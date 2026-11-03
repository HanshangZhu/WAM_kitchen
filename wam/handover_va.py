#!/usr/bin/env python
"""Hand-over in the closed loop: one model drives until its pot grasp fails, then another model takes over.

record   The first model runs as wam/eval_va.py does (same starts, seeds, calls in the same order). A pot-grasp
         attempt is a step where the commanded gripper turns from open to closed with the tcp within --near-xy of
         the pot (horizontally) and less than --near-z above it. If, --check steps later, the pot has risen less
         than --lift, the grasp has failed; the episode stops at the end of that chunk (the hand-over) and what
         was executed is saved: per chunk the env actions, which steps' frames went into the cache, and the
         executed-action array the cache received.
rescue   For each recorded episode: the same env and seed; per recorded chunk the new model is asked for a chunk
         (its output is discarded: this keeps the call order, the RNG and the cache layout of the original run),
         the recorded actions are stepped, and the same frames and executed actions go into the new model's
         cache. At the hand-over the new model drives, as in eval_va.py, until success or --max-steps in all.
         --fresh: the recorded actions are stepped without the model and the new model starts with an empty
         cache at the hand-over state (its first frame is the hand-over frame).
         With the first model itself as the "new" one, a rescue run must repeat the original eval episode:
         the check that the replay is exact.

  python wam/handover_va.py record --lora <no-noise>/lora_008000.pt --first 8 --episodes 24 --out <dir>
  python wam/handover_va.py rescue --lora <other>/lora_020000.pt --record <dir> --out <dir>/rescue_x [--fresh]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wam import _lingbot as LB  # noqa: E402
from wam.eval_va import CAM_KEY, SPLAT, make_config, merge_lora, pot_starts  # noqa: E402
from wam.render_va import H_RENDER, W, crop, with_wrist  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "rescue"])
    ap.add_argument("--lora", required=True)
    ap.add_argument("--record", help="rescue: the record mode's --out")
    ap.add_argument("--fresh", action="store_true", help="rescue: the new model starts with an empty cache")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--first", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--chunk", type=int, choices=[1, 2, 3, 4], default=4)
    ap.add_argument("--near-xy", type=float, default=0.10, help="m, tcp to pot, horizontal, at the closing step")
    ap.add_argument("--near-z", type=float, default=0.15, help="m, tcp above the pot's origin at most")
    ap.add_argument("--check", type=int, default=32, help="steps after the closing step")
    ap.add_argument("--lift", type=float, default=0.02, help="m the pot must have risen by then")
    ap.add_argument("--camera", default="take_030")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(os.path.dirname(a.lora), "meta.json")))
    L = LB.LAYOUTS["pretrain" if meta.get("slots", list(range(8))) == LB.LAYOUTS["pretrain"]["slots"] else "joints"]
    assert L["ee"] is not None, "ee_ik execution needs a --layout pretrain model"
    if a.mode == "rescue":
        recs = {int(k): v for k, v in json.load(open(os.path.join(a.record, "record.json")))["episodes"].items()}
        eps = sorted(k for k, v in recs.items() if v["handover_step"] is not None)
    else:
        eps = list(range(a.first, a.first + a.episodes))

    import gymnasium as gym
    import imageio
    import sapien
    import sim.views as V
    import wan_va.wan_va_server as S
    from modules.utils import WanVAEStreamingWrapper
    V.FRONT_W, V.FRONT_H = W, H_RENDER
    orig = V.camera_configs
    V.camera_configs = lambda env, take_=a.camera, hq=False: orig(env, take_, hq)
    import sim.microwave_env  # noqa: F401
    from sim.gs_client import SplatRenderer, composite

    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server"), a.chunk))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    from diffusers.video_processor import VideoProcessor
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    print(f"merged LoRA into {merge_lora(m.transformer, torch.load(a.lora, weights_only=False))} linears", flush=True)
    R = SplatRenderer(SPLAT)
    pots = pot_starts(max(eps) + 1)
    prompt = LB.PROMPTS["robot"]
    out_eps, results = {}, []

    for ep in eps:
        env = gym.make("MicrowavePot-v0", obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
                       pot_mw=(*pots[ep], np.pi / 2), views="hq", render_mode="rgb_array", sim_backend="cpu",
                       max_episode_steps=a.max_steps + 100)
        obs, _ = env.reset(seed=1000 + ep)
        u = env.unwrapped
        robot = u.agent.robot._objs[0]
        pm = robot.create_pinocchio_model()
        ee_link = [i for i, l in enumerate(robot.links) if l.name.endswith("panda_hand_tcp")][0]
        ids = [0, *u.ground.per_scene_id.tolist()]

        def front(o):
            img = crop(V.downsample(composite(o, R, "front", ids), "front"))
            if meta.get("wrist"):
                img = with_wrist(img, V.downsample(composite(o, R, "wrist", ids), "wrist"))
            return img

        def env_action(v):
            qj = np.asarray(v[L["joints"]], float)
            P = v[L["ee"]]
            q, ok, _ = pm.compute_inverse_kinematics(ee_link, sapien.Pose(P[:3], P[3:] / np.linalg.norm(P[3:])),
                                                     initial_qpos=np.r_[qj, 0.04, 0.04],
                                                     active_qmask=[1] * 7 + [0, 0], max_iterations=100)
            if ok and np.abs(q[:7] - qj).max() < 0.3:
                qj = q[:7]
            return np.r_[qj, v[L["grip"]]]

        def state():
            return dict(qpos=robot.get_qpos().tolist(), pot=u.pot.pose.raw_pose[0].tolist())

        torch.manual_seed(1000 + ep)
        m._reset(prompt)
        seen, lat, t, info, stop = [front(obs)], [], 0, {}, "max_steps"
        door_max, pot_zmax, ever_inside = 0.0, 0.0, False
        pot_z0 = float(u.pot.pose.p[0, 2])
        first, chunks, grip_prev = True, [], 1.0
        attempt, failed_at, handover_t, handover_frame, check_state = None, None, None, None, None
        rec = recs.get(ep) if a.mode == "rescue" else None
        replay = rec["chunks"] if rec else []

        def track():
            nonlocal door_max, pot_zmax, ever_inside
            door_max = max(door_max, float(info["door_deg"][0]))
            pot_zmax = max(pot_zmax, float(u.pot_in_mw()[0, 2]))
            ever_inside |= bool(info["pot_inside"][0])

        if rec and a.fresh:                          # the hand-over state without the model, then a fresh start
            for c in replay:
                for act, keep in zip(c["acts"], c["keep"]):
                    obs, _, _, _, info = env.step(torch.tensor(act).float()[None])
                    t += 1
                    track()
                    if keep:
                        seen.append(front(obs))
            handover_t, handover_frame, check_state = t, len(seen) - 1, state()
            replay = []
        ci = 0
        while t < a.max_steps:
            action, latents = m._infer({"obs": [{CAM_KEY: seen[-1]}]}, frame_st_id=m.frame_st_id)
            lat.append(latents)
            keys = []
            if ci < len(replay):                     # the recorded chunk, observed by this model
                c = replay[ci]
                for act, keep in zip(c["acts"], c["keep"]):
                    obs, _, _, _, info = env.step(torch.tensor(act).float()[None])
                    t += 1
                    track()
                    if keep:
                        f = front(obs)
                        keys.append({CAM_KEY: f})
                        seen.append(f)
                m._compute_kv_cache({"obs": keys, "state": np.asarray(c["done"], dtype=action.dtype)})
                first, ci = False, ci + 1
                if ci == len(replay):
                    handover_t, handover_frame, check_state = t, len(seen) - 1, state()
                continue
            done, acts, keep_l = action.copy(), [], []
            for i in range(1 if first else 0, action.shape[1]):
                for j in range(action.shape[2]):
                    act = env_action(action[:, i, j])
                    done[L["joints"], i, j] = act[:7]
                    pm.compute_forward_kinematics(np.r_[act[:7], 0.04, 0.04])
                    Pq = pm.get_link_pose(ee_link)
                    qq = np.asarray(Pq.q)
                    done[L["ee"], i, j] = np.r_[Pq.p, qq if qq @ action[L["ee"], i, j][3:] >= 0 else -qq]
                    obs, _, term, trunc, info = env.step(torch.from_numpy(act).float()[None])
                    t += 1
                    track()
                    acts.append(act.tolist())
                    keep = (j + 1) % 2 == 0
                    keep_l.append(keep)
                    if keep:
                        f = front(obs)
                        keys.append({CAM_KEY: f})
                        seen.append(f)
                    g = float(act[-1])
                    if a.mode == "record" and attempt is None and g < 0 <= grip_prev:
                        tcp, pot = u.agent.tcp.pose.p[0].numpy(), u.pot.pose.p[0].numpy()
                        dxy, dz = float(np.linalg.norm(tcp[:2] - pot[:2])), float(tcp[2] - pot[2])
                        if dxy < a.near_xy and dz < a.near_z:
                            attempt = dict(step=t, dxy_cm=round(dxy * 100, 1), dz_cm=round(dz * 100, 1))
                    grip_prev = g
                    if attempt is not None and failed_at is None and t == attempt["step"] + a.check:
                        rise = float(u.pot.pose.p[0, 2]) - pot_z0
                        attempt["rise_cm"] = round(rise * 100, 1)
                        if rise < a.lift:
                            failed_at = t
                    if bool(info.get("success", [False])[0]):
                        stop = "success"
                        break
                if stop == "success":
                    break
            first = False
            if stop == "success":
                break
            m._compute_kv_cache({"obs": keys, "state": done})
            chunks.append(dict(acts=acts, keep=keep_l, done=done.tolist()))
            if a.mode == "record" and failed_at is not None:
                handover_t, handover_frame, check_state = t, len(seen) - 1, state()
                stop = "handover"
                break

        ok = bool(info.get("success", [False])[0])
        r = dict(episode=ep, pot=pots[ep], steps=t, success=ok, stop=stop, door_max_deg=round(door_max, 1),
                 pot_zmax_cm=round(pot_zmax * 100, 1), pot_ever_inside=ever_inside, handover_step=handover_t,
                 handover_frame=handover_frame, attempt=attempt, chunks=len(lat), exec="ee_ik", chunk=a.chunk,
                 lora=a.lora, fresh=a.fresh if a.mode == "rescue" else None)
        if a.mode == "rescue":
            r["replay_exact"] = check_state == rec["state"]
            if not r["replay_exact"]:
                d = np.abs(np.array(check_state["qpos"]) - np.array(rec["state"]["qpos"])).max()
                r["replay_max_qpos_diff"] = float(d)
        else:
            out_eps[ep] = dict(handover_step=handover_t if stop == "handover" else None, attempt=attempt,
                               state=check_state, chunks=chunks if stop == "handover" else [])
        print(json.dumps(r), flush=True)
        results.append(r)
        imageio.mimwrite(os.path.join(a.out, f"ep{ep:02d}_seen.mp4"), seen, fps=10, codec="libx264", quality=7,
                         macro_block_size=1)
        if a.mode == "rescue":
            pv = m.decode_one_video(torch.cat(lat, 2).to(m.device), "np")[0]
            pv = (np.clip(pv, 0, 1) * 255).astype(np.uint8) if pv.dtype != np.uint8 else pv
            sv = seen[handover_frame:] if a.fresh else seen       # --fresh: imagination starts at the hand-over
            n = min(len(pv), len(sv))
            imageio.mimwrite(os.path.join(a.out, f"ep{ep:02d}_seen_vs_imagined.mp4"),
                             [np.vstack([sv[k], pv[k]]) for k in range(n)], fps=10, codec="libx264", quality=7,
                             macro_block_size=1)
        env.close()
        json.dump(results, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    if a.mode == "record":
        json.dump(dict(args=vars(a), episodes=out_eps), open(os.path.join(a.out, "record.json"), "w"))
    n = len(results)
    print(f"{a.mode}: {sum(x['success'] for x in results)} / {n} success"
          + (f", {sum(x['handover_step'] is not None and x['stop'] == 'handover' for x in results)} hand-overs"
             if a.mode == "record" else ""), flush=True)


if __name__ == "__main__":
    main()
