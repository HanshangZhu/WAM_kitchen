#!/usr/bin/env python
"""Closed-loop evaluation of a post-trained LingBot-VA (wam/train_va.py LoRA) on MicrowavePot-v0.

The loop is LingBot-VA's own (evaluation/libero/client.py): per chunk the model imagines 4 latent
frames of the front camera and the 32 pd_joint_pos actions that go with them; the actions are
executed (the first chunk skips frame 0's), every 2nd step's front image is kept (10 fps, 4 per
latent frame), and those frames + the executed actions go into the KV cache before the next chunk.

Front camera: the registered RealSense pose of --camera (a held-out take by default), 288 x 162 hq
-> 160 x 288 as in training, with the splatted kitchen behind the scene (--splat). Pot starts:
the same as sim/dit_vis.py eval's episodes 0..n-1 (seeds 1000 + i).

  python wam/eval_va.py --lora <dir>/lora_003000.pt --episodes 16 --out <dir>
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
from wam.render_va import H, H_RENDER, W, crop  # noqa: E402

SPLAT = os.path.join(ROOT, "Gitignore-but-important/recon/kitchen/gs_dense_po/ckpt.pt")
CAM_KEY = "observation.images.front"


def pot_starts(n, seed=1):
    gen = json.load(open(os.path.join(ROOT, "Gitignore-but-important/sim/gen/summary.json")))["runs"]
    pots = np.array([r["pot"] for r in gen])
    rng = np.random.default_rng(seed)
    return [rng.uniform(pots.min(0), pots.max(0)).round(4).tolist() for _ in range(n)]


def make_config(meta, guidance, save_root):
    from easydict import EasyDict
    from configs import VA_CONFIGS
    c = EasyDict(VA_CONFIGS["demo"])
    c.wan22_pretrained_model_name_or_path = LB.base_dir()
    c.height, c.width = H, W
    c.obs_cam_keys = [CAM_KEY]
    c.action_dim, c.action_per_frame, c.frame_chunk_size = 30, 8, 4
    slots = meta.get("slots", list(range(8)))         # LB.LAYOUTS; the first runs: 0..7
    c.used_action_channel_ids = slots
    inv = [len(slots)] * 30
    for i, j in enumerate(slots):
        inv[j] = i
    c.inverse_used_action_channel_ids = inv
    q01, q99 = [0.0] * 30, [0.0] * 30                 # per slot
    for i, j in enumerate(slots):
        q01[j], q99[j] = meta["q01"][i], meta["q99"][i]
    c.norm_stat = {"q01": q01, "q99": q99}
    c.snr_shift, c.action_snr_shift = 5.0, 1.0
    c.guidance_scale, c.action_guidance_scale = guidance, 1
    c.attn_window = int(os.environ.get("VA_ATTN_WINDOW", c.attn_window))   # KV cache: attn_window // 2 chunks
    c.enable_offload = os.environ.get("VA_OFFLOAD", "1") == "1"   # T5 on the CPU (the VAE is moved back to
                                       # the GPU below); VA_OFFLOAD=0 keeps T5 on the GPU (Spark: unified memory)
    c.rank = c.local_rank = 0
    c.world_size = 1
    c.env_type = "none"
    c.save_root = save_root
    return c


def merge_lora(transformer, sd, rank_alpha_ratio=1.0):
    """W += (alpha / r) B A into the plain nn.Linear weights; action layers loaded as trained"""
    params = dict(transformer.named_parameters())
    n = 0
    for k, A in sd.items():
        k0 = k.replace("_checkpoint_wrapped_module.", "")
        if k0.endswith(".A"):
            B = sd[k[:-2] + ".B"]
            w = params[k0[:-2] + ".weight"]
            w.data += (rank_alpha_ratio * (B.float() @ A.float())).to(w.dtype).to(w.device)
            n += 1
        elif k0.startswith(("action_embedder", "action_proj_out")):
            params[k0].data.copy_(A.to(params[k0].dtype))
    return n


@torch.no_grad()      # the server's infer() is no_grad; calling _infer / _compute_kv_cache directly is not
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", help="wam/train_va.py lora_*.pt (omit: the base checkpoint, actions meaningless)")
    ap.add_argument("--meta", help="the run's meta.json (default: next to --lora)")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=2600)
    ap.add_argument("--camera", default="take_030")
    ap.add_argument("--no-splat", action="store_true")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--prompt", default="robot", help="robot | human | any text")
    ap.add_argument("--exec", choices=["joints", "ee_ik"], default="joints",
                    help="joints: execute the predicted joint targets. ee_ik (a --layout pretrain model): the predicted "
                    "tcp pose, by IK seeded with the predicted joints (they pick the arm's configuration, the pose "
                    "where the hand goes); the joints themselves where the IK does not converge near them. "
                    "(ManiSkill's pd_ee_pose, IK from the current arm, cannot follow the demos' reconfigurations: "
                    "replaying a demo with it, the hand is 5 cm off from the first wrist turn on.)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(a.meta or os.path.join(os.path.dirname(a.lora), "meta.json")))
    L = LB.LAYOUTS["pretrain" if meta.get("slots", list(range(8))) == LB.LAYOUTS["pretrain"]["slots"] else "joints"]
    assert a.exec == "joints" or L["ee"] is not None, "--exec ee_ik needs a --layout pretrain model"
    ik_stats = dict(steps=0, used=0)

    def env_action(v, pm=None, ee=None):
        """one step of the model's action (used channels) -> pd_joint_pos action"""
        qj = np.asarray(v[L["joints"]], float)
        if a.exec == "ee_ik":
            import sapien
            P = v[L["ee"]]
            q, ok, _ = pm.compute_inverse_kinematics(ee, sapien.Pose(P[:3], P[3:] / np.linalg.norm(P[3:])),
                                                     initial_qpos=np.r_[qj, 0.04, 0.04],
                                                     active_qmask=[1] * 7 + [0, 0], max_iterations=100)
            ik_stats["steps"] += 1
            if ok and np.abs(q[:7] - qj).max() < 0.3:        # converged, on the predicted configuration
                qj = q[:7]
                ik_stats["used"] += 1
        return np.r_[qj, v[L["grip"]]]
    prompt = LB.PROMPTS.get(a.prompt, a.prompt)

    import gymnasium as gym
    import imageio
    import sim.views as V
    import wan_va.wan_va_server as S
    from modules.utils import WanVAEStreamingWrapper
    V.FRONT_W, V.FRONT_H = W, H_RENDER
    orig = V.camera_configs
    V.camera_configs = lambda env, take_=a.camera, hq=False: orig(env, take_, hq)
    import sim.microwave_env  # noqa: F401
    from sim.gs_client import SplatRenderer, composite

    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server")))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    from diffusers.video_processor import VideoProcessor
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    if a.lora:
        sd = torch.load(a.lora, weights_only=False)
        print(f"merged LoRA into {merge_lora(m.transformer, sd)} linears", flush=True)
    R = None if a.no_splat else SplatRenderer(SPLAT)
    pots = pot_starts(a.first + a.episodes)
    results = []
    for ep in range(a.first, a.first + a.episodes):
        env = gym.make("MicrowavePot-v0", obs_mode="rgb+segmentation", control_mode="pd_joint_pos",
                       pot_mw=(*pots[ep], np.pi / 2), views="hq", render_mode="rgb_array", sim_backend="cpu",
                       max_episode_steps=a.max_steps + 100)
        obs, _ = env.reset(seed=1000 + ep)
        u = env.unwrapped
        robot = u.agent.robot._objs[0]
        pm = robot.create_pinocchio_model() if a.exec == "ee_ik" else None
        ee_link = [i for i, l in enumerate(robot.links) if l.name.endswith("panda_hand_tcp")][0]
        ids = [0, *u.ground.per_scene_id.tolist()]

        def front(o):
            img = composite(o, R, "front", ids) if R else o["sensor_data"]["front"]["rgb"][0].cpu().numpy()
            return crop(V.downsample(img, "front"))

        torch.manual_seed(1000 + ep)
        m._reset(prompt)
        seen, lat, t0, t, info, stop = [front(obs)], [], time.time(), 0, {}, "max_steps"
        door_max, pot_zmax, ever_inside = 0.0, 0.0, False
        first = True
        while t < a.max_steps:
            action, latents = m._infer({"obs": [{CAM_KEY: seen[-1]}]}, frame_st_id=m.frame_st_id)
            lat.append(latents)
            keys = []
            for i in range(1 if first else 0, action.shape[1]):
                for j in range(action.shape[2]):
                    obs, _, term, trunc, info = env.step(torch.from_numpy(env_action(action[:, i, j], pm, ee_link)).float()[None])
                    t += 1
                    door_max = max(door_max, float(info["door_deg"][0]))
                    pot_zmax = max(pot_zmax, float(u.pot_in_mw()[0, 2]))
                    ever_inside |= bool(info["pot_inside"][0])
                    if (j + 1) % 2 == 0:
                        f = front(obs)
                        keys.append({CAM_KEY: f})
                        seen.append(f)
                    if bool(info.get("success", [False])[0]):
                        stop = "success"
                        break
                if stop == "success":
                    break
            first = False
            if stop == "success":
                break
            m._compute_kv_cache({"obs": keys, "state": action})
        ok = bool(info.get("success", [False])[0])
        r = dict(episode=ep, pot=pots[ep], steps=t, success=ok, stop=stop, door_max_deg=round(door_max, 1),
                 pot_zmax_cm=round(pot_zmax * 100, 1), pot_ever_inside=ever_inside, wall_s=round(time.time() - t0),
                 chunks=len(lat), camera=a.camera, splat=R is not None, exec=a.exec,
                 ik_used=round(ik_stats["used"] / max(ik_stats["steps"], 1), 3))
        ik_stats.update(steps=0, used=0)
        print(json.dumps(r), flush=True)
        results.append(r)
        imageio.mimwrite(os.path.join(a.out, f"ep{ep:02d}_seen.mp4"), seen, fps=10, codec="libx264", quality=7,
                         macro_block_size=1)
        try:
            with torch.no_grad():
                pv = m.decode_one_video(torch.cat(lat, 2).to(m.device), "np")[0]
            pv = (np.clip(pv, 0, 1) * 255).astype(np.uint8) if pv.dtype != np.uint8 else pv
            n = min(len(pv), len(seen))
            imageio.mimwrite(os.path.join(a.out, f"ep{ep:02d}_seen_vs_imagined.mp4"),
                             [np.vstack([seen[k], pv[k]]) for k in range(n)], fps=10, codec="libx264", quality=7,
                             macro_block_size=1)
        except Exception as e:                                      # noqa: BLE001
            print("decode failed:", e, flush=True)
        env.close()
        json.dump(results, open(os.path.join(a.out, f"summary_{a.first}-{a.first + a.episodes - 1}.json"), "w"),
                  indent=1)
    print(f"success {sum(r['success'] for r in results)} / {len(results)}", flush=True)


if __name__ == "__main__":
    main()
