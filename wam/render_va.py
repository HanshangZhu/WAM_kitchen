#!/usr/bin/env python
"""Front-camera videos of the solver demonstrations for LingBot-VA post-training.

Each successful run's simulator states are re-rendered (no re-simulation, as sim/render_views.py)
from the registered RealSense pose of one of the human takes -- a different training take per run,
so the robot data sees the same spread of camera poses as the ego video -- at 10 fps (every 2nd
20 Hz step), hq shader, 288 x 162 (the D435's 16:9), rows 1..160 kept -> 160 x 288.

  python wam/render_va.py --data <gen dir>[,...] --takes take_003,take_004,... [--jobs 8] [--splat <ckpt>]

--splat: the splatted kitchen behind the robot, microwave and pot (composited by segmentation
at the render resolution, as sim/gs_client.py), so robot and ego video share room and viewpoint.

Writes <run>/va_front.npz: idx (env step of each frame), front (N, 160, 288, 3) uint8, take.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

W, H, H_RENDER = 288, 160, 162
NAME = "va_front"


def crop(img):
    """(162, 288, 3) -> (160, 288, 3): the 16:9 frame with one row off top and bottom"""
    return img[1:1 + H]


WRIST_SIDE = 160    # the hand camera (128 x 128) scaled to the front frame's height, to its right


def with_wrist(front, wrist):
    """front (160, 288, 3) + wrist (128, 128, 3) -> (160, 448, 3): the two views side by side, encoded and
    predicted as one frame (LingBot-VA then sees one camera; nothing in its code changes)"""
    import cv2
    return np.concatenate([front, cv2.resize(wrist, (WRIST_SIDE, WRIST_SIDE), interpolation=cv2.INTER_AREA)], 1)


def render_run(args):
    run_dir, stride, take, splat, suffix = args[:5]
    wrist = len(args) > 5 and args[5]
    out = os.path.join(run_dir, NAME + ("_splat" if splat else "") + suffix + ".npz")
    if os.path.exists(out):
        return run_dir, "exists"
    import torch
    import gymnasium as gym
    import sim.views as V
    V.FRONT_W, V.FRONT_H = W, H_RENDER
    orig = V.camera_configs
    V.camera_configs = lambda env, take_=take, hq=False: orig(env, take_, hq)
    import sim.microwave_env  # noqa: F401
    from sim.render_views import states_of
    torch.set_num_threads(1)
    env = gym.make("MicrowavePot-v0", obs_mode="rgb+segmentation" if splat else "rgb", control_mode="pd_joint_pos",
                   sim_backend="cpu", views="hq")
    env.reset(seed=0)
    u = env.unwrapped
    if splat:
        from sim.gs_client import SplatRenderer, composite
        R = SplatRenderer(splat)
        ids = [0, *u.ground.per_scene_id.tolist()]
    kind, S = states_of(run_dir)
    n_obs = len(np.load(os.path.join(run_dir, "demo.npz"))["obs"]) if os.path.exists(os.path.join(run_dir, "demo.npz")) \
        else len(S["articulations"]["panda"]) - 1
    idx = np.arange(0, n_obs, stride)
    front, wr = [], []
    for i in idx:
        if kind == "flat":
            u.set_state(torch.tensor(S[i:i + 1]))
        else:
            u.set_state_dict({"actors": {k: torch.tensor(v[i:i + 1]) for k, v in S["actors"].items()},
                              "articulations": {k: torch.tensor(v[i:i + 1]) for k, v in S["articulations"].items()}})
        obs = u.get_obs()
        img = composite(obs, R, "front", ids) if splat else obs["sensor_data"]["front"]["rgb"][0].cpu().numpy()
        front.append(crop(V.downsample(img, "front")))
        if wrist:
            wimg = composite(obs, R, "wrist", ids) if splat else obs["sensor_data"]["wrist"]["rgb"][0].cpu().numpy()
            wr.append(V.downsample(wimg, "wrist"))
    extra = dict(wrist=np.stack(wr)) if wrist else {}
    np.savez_compressed(out, idx=idx, front=np.stack(front), take=take, **extra)
    if splat:
        R.close()
    env.close()
    return run_dir, f"{len(idx)} frames, camera {take}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--takes", required=True, help="comma-separated takes whose camera poses are used in turn")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--splat", help="splat checkpoint (recon/gs_kitchen.py): the kitchen behind the scene")
    ap.add_argument("--limit", type=int, default=0, help="first N successful runs only (smoke test)")
    ap.add_argument("--wrist", action="store_true", help="also the hand camera (128 x 128), splatted like the front")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _back03 (with FRONT_CAM_BACK / FRONT_CAM_UP set)")
    a = ap.parse_args()
    runs = []
    for d in a.data.split(","):
        s = json.load(open(os.path.join(d, "summary.json")))
        runs += [os.path.join(d, f"run_{r['run']:03d}") for r in s["runs"] if r["success"]]
    if a.limit:
        runs = runs[:a.limit]
    takes = a.takes.split(",")
    print(f"{len(runs)} successful runs, {len(takes)} camera poses", flush=True)
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        jobs = [(r, a.stride, takes[k % len(takes)], a.splat, a.suffix, a.wrist) for k, r in enumerate(runs)]
        for k, (r, msg) in enumerate(ex.map(render_run, jobs)):
            print(f"  {k + 1:3d}/{len(runs)} {os.path.relpath(r, ROOT)}: {msg}", flush=True)


if __name__ == "__main__":
    main()
