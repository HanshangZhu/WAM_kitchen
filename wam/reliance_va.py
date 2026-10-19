#!/usr/bin/env python
"""How much do a --layout pretrain model's predicted actions depend on the action history, and how much on the
video? Two measurements on held-out solver demos, teacher-forced (the true past goes into the cache, as in
wam/action_acc_va.py):

1. Intervention. The demo is walked four times, each with its own cache:
     orig        the true frames and the true actions go into the cache
     seed        the same, another sampling seed: how much the prediction moves by sampling alone
     act_shift   the true frames; the actions written into the cache are those of SHIFT chunks later
     vid_shift   the true actions; the frames written into the cache (and the current frame) are SHIFT chunks later
   Per chunk, the predicted motion is compared with orig's: the mean distance between FK(predicted joints)
   (tcp, cm). act_shift moving the prediction much more than vid_shift = the actions follow the action history.

2. Conflict. Two demos reach a tuck at nearly the same step with nearly the same past motion, and differ in
   what comes next only through what the scene shows: gen_retry (the pot left on the cavity's edge -> go back to
   the pot) and gen_s2 (the pot in -> go to the door's handle). After the tuck, where does the predicted tcp go:
   its distance to the pot and to the handle at the end of each of the next chunks.

  python wam/reliance_va.py --lora <run>/lora_001500.pt --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wam import _lingbot as LB  # noqa: E402
from wam.encode_va import robot_actions  # noqa: E402
from wam.eval_va import CAM_KEY, make_config, merge_lora  # noqa: E402
from wam.prefix_va import CHUNK, FIRST, chunk_actions  # noqa: E402
from wam.head_consistency_va import kinematics  # noqa: E402

G = "Gitignore-but-important/sim"
SHIFT = 2


def targets(run):
    """mw-frame pot grasp point and door-handle point at the tuck after the place (world coordinates)"""
    import gymnasium as gym
    import sim.microwave_env as M
    from sim.solve_microwave import Solver, POT_GRASP_Z
    r = json.load(open(os.path.join(run, "result.json")))
    if r.get("place_short") is not None:      # the tuck after the failed place
        dec = [p for p in r["phases"] if p["phase"] == "left at the opening (not inside)"][0]
    else:                                     # the tuck after the place (the second tuck)
        dec = [p for p in r["phases"] if p["phase"] == "tuck"][1]
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_backend="none", door_q0=np.deg2rad(dec["door_deg"]))
    env.reset(seed=0)
    S = Solver(env, None)
    pot = M.mw_to_world(np.r_[dec["pot_mw"][:2], dec["pot_mw"][2] + POT_GRASP_Z])
    handle = np.asarray(S.handle_pose(np.deg2rad(dec["door_deg"])).p)
    env.close()
    return dec["t"], pot, handle


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--video", default="va_front_splat_back03")
    ap.add_argument("--runs", default=f"{G}/gen_s2/run_011,{G}/gen_stage/run_012,{G}/gen_retry/run_015")
    ap.add_argument("--conflict", default=f"{G}/gen_retry/run_015,{G}/gen_s2/run_011")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(os.path.dirname(a.lora), "meta.json")))
    L = LB.LAYOUTS["pretrain"]
    assert meta.get("slots") == L["slots"], "needs a --layout pretrain run"
    fk, _ = kinematics()

    import wan_va.wan_va_server as S
    from diffusers.video_processor import VideoProcessor
    from modules.utils import WanVAEStreamingWrapper
    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server")))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    print(f"merged LoRA into {merge_lora(m.transformer, torch.load(a.lora, weights_only=False))} linears", flush=True)
    m._reset(LB.PROMPTS["robot"])
    emb = m.prompt_embeds, m.negative_prompt_embeds

    def state15(t8):
        out = np.zeros((15, 4, 8), np.float32)
        for f in range(4):
            for s in range(8):
                q = t8[:7, f, s]
                if np.any(q):
                    out[L["ee"], f, s] = fk(q)
                out[L["joints"], f, s] = q
                out[L["grip"], f, s] = t8[7, f, s]
        return out

    def walk(run, cond, seed=0, stop_chunk=None):
        """predicted tcp positions (n_chunks, steps, 3) from FK of the predicted joints"""
        z = np.load(os.path.join(run, a.video + ".npz"))
        front, idx = z["front"], z["idx"]
        acts = robot_actions(run)
        n = 1 + (len(front) - 1 - FIRST) // CHUNK
        n = n if stop_chunk is None else min(n, stop_chunk)
        torch.manual_seed(seed)
        m._reset(None)
        m.prompt_embeds, m.negative_prompt_embeds = emb
        out = []
        for c in range(n):
            lo = 0 if c == 0 else FIRST + CHUNK * (c - 1)
            sh = SHIFT if cond == "vid_shift" else 0
            lo_v = min(lo + sh * CHUNK, len(front) - 1 - (FIRST if c == 0 else CHUNK))
            pred, _ = m._infer({"obs": [{CAM_KEY: front[lo_v]}]}, frame_st_id=m.frame_st_id)
            f0 = 1 if c == 0 else 0
            P = pred[:, f0:].reshape(15, -1).T
            out.append(np.array([fk(p[L["joints"]])[:3] for p in P]))
            ca = min(c + SHIFT, n - 1) if cond == "act_shift" else c
            true = state15(chunk_actions(acts, idx, ca))
            keys = [{CAM_KEY: front[k]} for k in range(lo_v + 1, lo_v + 1 + (FIRST if c == 0 else CHUNK))]
            m._compute_kv_cache({"obs": keys, "state": true})
        return out, idx

    report = dict(shift_chunks=SHIFT, intervention={}, conflict={})
    for run in a.runs.split(","):
        P = {cond: walk(run, cond, seed=1 if cond == "seed" else 0)[0] for cond in ("orig", "seed", "act_shift", "vid_shift")}
        n = min(len(v) for v in P.values())
        d = {cond: [float(np.mean(np.linalg.norm(P[cond][c][:min(len(P[cond][c]), len(P["orig"][c]))] -
                                                P["orig"][c][:min(len(P[cond][c]), len(P["orig"][c]))], axis=1)) * 100)
                    for c in range(n)] for cond in ("seed", "act_shift", "vid_shift")}
        summ = {cond: dict(median_cm=round(float(np.median(v)), 2), mean_cm=round(float(np.mean(v)), 2)) for cond, v in d.items()}
        summ["reliance_ratio_act_over_vid"] = round(float(np.mean(d["act_shift"]) / max(np.mean(d["vid_shift"]), 1e-6)), 2)
        report["intervention"][run] = dict(summary=summ, per_chunk_cm=d)
        print("INTERVENTION", os.path.basename(os.path.dirname(run)), os.path.basename(run), json.dumps(summ), flush=True)

    for run in a.conflict.split(","):
        t_dec, pot, handle = targets(run)
        z = np.load(os.path.join(run, a.video + ".npz"))
        idx = z["idx"]
        # the first chunk whose start frame is at or after the decision step
        c_dec = next(c for c in range(1, 1000) if idx[min(FIRST + CHUNK * (c - 1), len(idx) - 1)] >= t_dec)
        rows = []
        for seed in (0, 1, 2):
            P, _ = walk(run, "orig", seed=seed, stop_chunk=c_dec + 3)
            for k, c in enumerate(range(c_dec, min(c_dec + 3, len(P)))):
                end = P[c][-1]
                rows.append(dict(seed=seed, chunk_after_decision=k, d_pot_cm=round(float(np.linalg.norm(end - pot) * 100), 1),
                                 d_handle_cm=round(float(np.linalg.norm(end - handle) * 100), 1)))
        start = P[c_dec - 1][-1]
        report["conflict"][run] = dict(decision_step=int(t_dec), decision_chunk=int(c_dec),
                                       at_decision=dict(d_pot_cm=round(float(np.linalg.norm(start - pot) * 100), 1),
                                                        d_handle_cm=round(float(np.linalg.norm(start - handle) * 100), 1)),
                                       next_chunks=rows)
        print("CONFLICT", os.path.basename(os.path.dirname(run)), os.path.basename(run), json.dumps(report["conflict"][run]), flush=True)
    json.dump(report, open(os.path.join(a.out, "reliance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
