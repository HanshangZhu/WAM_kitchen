#!/usr/bin/env python
"""How close are a post-trained LingBot-VA's actions to a held-out demonstration's, with the true past?

Walks a solver demonstration chunk by chunk as the closed loop does (wam/eval_va.py): per chunk the
model predicts its 32 actions from the cache, the prediction is compared with the demo's actions for
that chunk, then the demo's real frames and actions go into the cache. So every prediction has the
true history (teacher forcing); errors are per chunk, not compounded.

  python wam/action_acc_va.py --lora <run>/lora_000250.pt --runs <gen>/run_011,... [--video va_front_splat_back03]

Prints per run: joint error (rad, mean over the 7 arm joints) and tcp position error (cm, by forward
kinematics) for the chunk's first 8 actions and for all 32, against time in the demo; writes
<out>/action_acc.json.
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


def tcp_fk():
    """q (7,) -> tcp position (3,) in metres, the Panda of MicrowavePot-v0"""
    import gymnasium as gym
    import sim.microwave_env  # noqa: F401
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_backend="none")
    env.reset(seed=0)
    u = env.unwrapped
    robot = u.agent.robot

    def f(q):
        qp = robot.get_qpos().clone()
        qp[0, :7] = torch.as_tensor(q, dtype=qp.dtype)
        robot.set_qpos(qp)
        return u.agent.tcp.pose.p[0].numpy().copy()
    return f


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated held-out solver run dirs")
    ap.add_argument("--video", default="va_front_splat", help="the run's front video npz (e.g. va_front_splat_back03)")
    ap.add_argument("--prompt", default="robot")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(os.path.dirname(a.lora), "meta.json")))
    fk = tcp_fk()

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
            pred, _ = m._infer({"obs": [{CAM_KEY: front[lo]}]}, frame_st_id=m.frame_st_id)   # (8, 4, 8) raw
            true = chunk_actions(acts, idx, c)
            f0 = 1 if c == 0 else 0                                    # latent frame 0 of chunk 0 has no actions
            p = pred[:, f0:].reshape(8, -1).T                          # (steps, 8)
            t = true[:, f0:].reshape(8, -1).T
            jerr = np.abs(p[:, :7] - t[:, :7]).mean(1)
            perr = np.array([np.linalg.norm(fk(pp[:7]) - fk(tt[:7])) for pp, tt in zip(p[::4], t[::4])]) * 100
            grip = float(np.mean(np.sign(p[:, 7]) != np.sign(t[:, 7])))
            rows.append(dict(t=round(lo / 10, 1), joint_rad_first8=float(jerr[:8].mean()), joint_rad=float(jerr.mean()),
                             tcp_cm_first8=float(perr[:2].mean()), tcp_cm=float(perr.mean()), gripper_wrong=grip))
            keys = [{CAM_KEY: front[k]} for k in range(lo + 1, lo + 1 + (FIRST if c == 0 else CHUNK))]
            m._compute_kv_cache({"obs": keys, "state": true})
        report[run] = rows
        r = np.array([[x["tcp_cm_first8"], x["tcp_cm"], x["joint_rad"], x["gripper_wrong"]] for x in rows])
        print(f"{os.path.basename(run)}: {len(rows)} chunks; tcp error first 0.4 s {r[:, 0].mean():.1f} cm "
              f"(median {np.median(r[:, 0]):.1f}), whole chunk {r[:, 1].mean():.1f} cm; joint {r[:, 2].mean():.3f} rad; "
              f"gripper sign wrong {r[:, 3].mean() * 100:.0f}%", flush=True)
        for x in rows[::3]:
            print(f"   t {x['t']:5.1f}s  tcp {x['tcp_cm_first8']:5.1f} / {x['tcp_cm']:5.1f} cm  joint {x['joint_rad']:.3f}  "
                  f"grip wrong {x['gripper_wrong']:.2f}", flush=True)
    json.dump(report, open(os.path.join(a.out, "action_acc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
