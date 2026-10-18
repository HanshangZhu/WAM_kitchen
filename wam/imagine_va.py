#!/usr/bin/env python
"""Pure imagination (LingBot-VA's i2va: a first frame + the prompt, N chunks, its own predictions fed
back, nothing executed) with a post-trained LoRA, from the first frames of held-out takes (real) and
of the evaluation's simulated start states (splatted kitchen, robot in view).

  python wam/imagine_va.py --lora <run>/lora_002500.pt --out <dir> [--chunks 20] [--seeds 2]

Writes <out>/<input>_s<seed>.mp4 (the imagined front video) and a contact sheet <out>/sheet.png.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wam import _lingbot as LB  # noqa: E402
from wam.encode_va import human_frames  # noqa: E402
from wam.eval_va import CAM_KEY, SPLAT, make_config, merge_lora, pot_starts  # noqa: E402
from wam.render_va import H_RENDER, W, crop  # noqa: E402


def sim_first_frame(ep, camera):
    import gymnasium as gym
    import sim.views as V
    V.FRONT_W, V.FRONT_H = W, H_RENDER
    orig = V.camera_configs
    V.camera_configs = lambda env, take_=camera, hq=False: orig(env, take_, hq)
    import sim.microwave_env  # noqa: F401
    from sim.gs_client import SplatRenderer, composite
    pot = pot_starts(ep + 1)[ep]
    env = gym.make("MicrowavePot-v0", obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
                   pot_mw=(*pot, np.pi / 2), views="hq", sim_backend="cpu")
    obs, _ = env.reset(seed=1000 + ep)
    R = SplatRenderer(SPLAT)
    img = crop(V.downsample(composite(obs, R, "front", [0, *env.unwrapped.ground.per_scene_id.tolist()]), "front"))
    R.close()
    env.close()
    V.camera_configs = orig
    return img


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", help="omit: the base checkpoint")
    ap.add_argument("--meta", help="a run's meta.json (needed without --lora)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--takes", default="take_027,take_035,take_041")
    ap.add_argument("--sim-episodes", default="0,1")
    ap.add_argument("--camera", default="take_030")
    ap.add_argument("--prompt", default="robot")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--first-frames", help="npz {take: first frame (160, 288, 3)} instead of reading the takes' videos")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(a.meta or os.path.join(os.path.dirname(a.lora), "meta.json")))
    if a.first_frames:
        z = np.load(a.first_frames)
        inputs = {t: z[t] for t in a.takes.split(",") if t}
    else:
        inputs = {t: human_frames(t)[0] for t in a.takes.split(",") if t}
    for e in [int(x) for x in a.sim_episodes.split(",") if x]:
        inputs[f"sim_ep{e}"] = sim_first_frame(e, a.camera)

    import imageio
    import wan_va.wan_va_server as S
    from diffusers.video_processor import VideoProcessor
    from modules.utils import WanVAEStreamingWrapper
    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server")))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    if a.lora:
        print(f"merged LoRA into {merge_lora(m.transformer, torch.load(a.lora, weights_only=False))} linears", flush=True)
    rows = []
    for name, img in inputs.items():
        for s in range(a.seeds):
            torch.manual_seed(s)
            m._reset(LB.PROMPTS.get(a.prompt, a.prompt))
            lat = []
            for k in range(a.chunks):
                _, l = m._infer({"obs": [{CAM_KEY: img}]}, frame_st_id=k * 4)
                lat.append(l)
            v = m.decode_one_video(torch.cat(lat, 2), "np")[0]
            v = (np.clip(v, 0, 1) * 255).astype(np.uint8) if v.dtype != np.uint8 else v
            imageio.mimwrite(os.path.join(a.out, f"{name}_s{s}.mp4"), v, fps=10, codec="libx264", quality=7,
                             macro_block_size=1)
            idx = np.linspace(0, len(v) - 1, 8).astype(int)
            row = np.ascontiguousarray(np.hstack([v[i] for i in idx]), dtype=np.uint8).copy()
            cv2.putText(row, f"{name} s{s}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            rows.append(row)
            print(name, s, v.shape, flush=True)
    cv2.imwrite(os.path.join(a.out, "sheet.png"), np.vstack(rows)[..., ::-1])


if __name__ == "__main__":
    main()
