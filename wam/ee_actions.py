#!/usr/bin/env python
"""The tcp pose of every robot clip's joint targets, for the pretraining's channel layout.

LingBot-VA was pretrained with the end-effector pose in action slots 0..6 (wan_va/configs/
va_franka_cfg.py); wam/encode_va.py stores joint targets. For each <data>/robot/<clip>.pt this
writes <clip>.ee.pt: (F, 8, 7) float32, the tcp pose (xyz, then a wxyz quaternion, robot base
frame = world) that the clip's joint targets put the hand at, by the Panda's forward kinematics.
The quaternion's sign is kept continuous along the clip (q and -q are the same rotation; a jump
between them would be a jump in the channel). Row 0 (no actions) is left zero.

  python wam/ee_actions.py --data Gitignore-but-important/wam/latents_stage
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fk_fn():
    """q (7,) -> tcp pose (7,): xyz, wxyz"""
    import gymnasium as gym
    import sim.microwave_env  # noqa: F401
    env = gym.make("MicrowavePot-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   render_backend="none")
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot._objs[0]
    pm = robot.create_pinocchio_model()
    ee = [i for i, l in enumerate(robot.links) if l.name.endswith("panda_hand_tcp")][0]
    full = np.zeros(9)

    def f(q):
        full[:7] = q
        pm.compute_forward_kinematics(full)
        P = pm.get_link_pose(ee)
        return np.r_[P.p, P.q]
    return f


def tcp_poses(joints, fk):
    """joints (N, 7) -> (N, 7), the quaternion sign continuous (the first with w >= 0)"""
    out = np.array([fk(q) for q in joints], np.float32)
    for i in range(len(out)):
        ref = out[i - 1, 3:] if i else np.array([1.0, 0, 0, 0])
        if out[i, 3:] @ ref < 0:
            out[i, 3:] *= -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="an encode_va.py output dir (its robot/ clips)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fk = fk_fn()
    files = sorted(f for f in glob.glob(os.path.join(a.data, "robot", "*.pt")) if not f.endswith(".ee.pt"))
    for i, f in enumerate(files):
        out = f[:-3] + ".ee.pt"
        if os.path.exists(out) and not a.force:
            continue
        act = torch.load(f, weights_only=False)["actions"].numpy()         # (F, 8, 8): joints 7 + gripper
        Fn, n, _ = act.shape
        ee = np.zeros((Fn, n, 7), np.float32)
        ee[1:] = tcp_poses(act[1:, :, :7].reshape(-1, 7), fk).reshape(Fn - 1, n, 7)
        torch.save(torch.from_numpy(ee), out)
        if i % 10 == 0:
            print(f"{i + 1}/{len(files)} {os.path.basename(f)}: tcp x {ee[1:, :, 0].min():.2f}..{ee[1:, :, 0].max():.2f}",
                  flush=True)
    print(f"{len(files)} clips", flush=True)


if __name__ == "__main__":
    main()
