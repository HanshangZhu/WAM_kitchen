#!/usr/bin/env python
"""For a --layout pretrain model: are its two action outputs, the tcp pose and the joint targets, the same motion,
and which of them is closer to the demonstration?

Walks held-out solver demos chunk by chunk with the true past in the cache (teacher forcing, as
wam/action_acc_va.py). Per chunk and step it compares
  pose vs joints   the predicted tcp pose against FK(predicted joints)          -- do the two outputs agree?
  pose vs demo     the predicted tcp pose against FK(the demo's joint target)   -- the pose output's error
  joints vs demo   FK(predicted joints) against FK(the demo's joint target)      -- the joint output's error
and runs the --exec ee_ik step (pinocchio IK to the predicted pose, seeded with the predicted joints): how often
it converges, and how far its answer is from the seed.

  python wam/head_consistency_va.py --lora <run>/lora_000250.pt --runs <gen>/run_011,... \
      --video va_front_splat_back03 --out <dir>
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


def kinematics():
    import gymnasium as gym
    import sapien
    import sim.microwave_env  # noqa: F401
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_backend="none")
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot._objs[0]
    pm = robot.create_pinocchio_model()
    ee = [i for i, l in enumerate(robot.links) if l.name.endswith("panda_hand_tcp")][0]

    def fk(q):
        pm.compute_forward_kinematics(np.r_[q, 0.04, 0.04])
        P = pm.get_link_pose(ee)
        return np.r_[P.p, P.q]

    def ik(pose, seed):
        q, ok, _ = pm.compute_inverse_kinematics(ee, sapien.Pose(pose[:3], pose[3:] / np.linalg.norm(pose[3:])),
                                                 initial_qpos=np.r_[seed, 0.04, 0.04], active_qmask=[1] * 7 + [0, 0],
                                                 max_iterations=100)
        return bool(ok), q[:7]
    return fk, ik


def pose_err(a, b):
    """position (cm), rotation (deg) between two (xyz, wxyz) poses"""
    d = np.linalg.norm(a[:3] - b[:3]) * 100
    qa, qb = a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:])
    return float(d), float(np.degrees(2 * np.arccos(min(1.0, abs(float(qa @ qb))))))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--video", default="va_front_splat_back03")
    ap.add_argument("--prompt", default="robot")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(os.path.dirname(a.lora), "meta.json")))
    L = LB.LAYOUTS["pretrain"]
    assert meta.get("slots") == L["slots"], "needs a --layout pretrain run"
    fk, ik = kinematics()

    import wan_va.wan_va_server as S
    from diffusers.video_processor import VideoProcessor
    from modules.utils import WanVAEStreamingWrapper
    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server")))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    print(f"merged LoRA into {merge_lora(m.transformer, torch.load(a.lora, weights_only=False))} linears", flush=True)
    m._reset(LB.PROMPTS.get(a.prompt, a.prompt))
    emb = m.prompt_embeds, m.negative_prompt_embeds

    report = {}
    for run in a.runs.split(","):
        z = np.load(os.path.join(run, a.video + ".npz"))
        front, idx = z["front"], z["idx"]
        acts = robot_actions(run)
        n_chunks = 1 + (len(front) - 1 - FIRST) // CHUNK
        torch.manual_seed(0)
        m._reset(None)
        m.prompt_embeds, m.negative_prompt_embeds = emb
        rows = []
        for c in range(n_chunks):
            lo = 0 if c == 0 else FIRST + CHUNK * (c - 1)
            pred, _ = m._infer({"obs": [{CAM_KEY: front[lo]}]}, frame_st_id=m.frame_st_id)      # (15, 4, 8)
            t8 = chunk_actions(acts, idx, c)                                                   # (8, 4, 8)
            f0 = 1 if c == 0 else 0
            P = pred[:, f0:].reshape(15, -1).T                                                 # (steps, 15)
            T = t8[:, f0:].reshape(8, -1).T
            for k in range(0, len(P), 4):
                pe, pj, tj = P[k, L["ee"]], P[k, L["joints"]], T[k, :7]
                f_pj, f_tj = fk(pj), fk(tj)
                ok, qik = ik(pe, pj)
                rows.append(dict(t=round((lo + k / 2) / 10, 1),
                                 pose_vs_joints=pose_err(pe, f_pj), pose_vs_demo=pose_err(pe, f_tj),
                                 joints_vs_demo=pose_err(f_pj, f_tj), ik_ok=ok,
                                 ik_dq_max=float(np.abs(qik - pj).max())))
            # the true past, in the run's layout: tcp pose of the demo's targets, joints, gripper
            true = np.zeros((15, 4, 8), np.float32)
            for f in range(4):
                for s in range(8):
                    q = t8[:7, f, s]
                    if np.any(q):
                        true[L["ee"], f, s] = fk(q)
                    true[L["joints"], f, s] = q
                    true[L["grip"], f, s] = t8[7, f, s]
            keys = [{CAM_KEY: front[k]} for k in range(lo + 1, lo + 1 + (FIRST if c == 0 else CHUNK))]
            m._compute_kv_cache({"obs": keys, "state": true})
        r = rows
        med = lambda key, i: float(np.median([x[key][i] for x in r]))
        p90 = lambda key, i: float(np.percentile([x[key][i] for x in r], 90))
        summ = {k: dict(pos_cm_median=round(med(k, 0), 2), pos_cm_p90=round(p90(k, 0), 2),
                        rot_deg_median=round(med(k, 1), 1), rot_deg_p90=round(p90(k, 1), 1))
                for k in ("pose_vs_joints", "pose_vs_demo", "joints_vs_demo")}
        summ["ik_ok"] = round(float(np.mean([x["ik_ok"] for x in r])), 3)
        summ["ik_ok_and_within_0.3rad"] = round(float(np.mean([x["ik_ok"] and x["ik_dq_max"] < 0.3 for x in r])), 3)
        summ["ik_dq_max_median"] = round(float(np.median([x["ik_dq_max"] for x in r])), 3)
        report[run] = dict(summary=summ, rows=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in x.items()} for x in r])
        print(os.path.basename(run), json.dumps(summ), flush=True)
    json.dump(report, open(os.path.join(a.out, "head_consistency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
