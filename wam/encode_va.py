#!/usr/bin/env python
"""Wan2.2-VAE latents of the training clips for LingBot-VA post-training (one front camera).

Robot clips: the solver demonstrations' front video (wam/render_va.py, 10 fps) with their
pd_joint_pos actions (20 Hz, 8-d: 7 joints + gripper), 8 actions per latent frame.
Human clips: the ego takes (30 fps RealSense .avi -> every 3rd frame), the same 160 x 288
crop of the same 16:9 camera, no actions.

A clip is 1 + 4 K frames (K latent frames after the first); clips start every --hop frames.
Frames are encoded as the server encodes observations (the first frame alone, then 16-frame
chunks through the streaming VAE), normalised with the VAE's latent mean / std.

  python wam/encode_va.py --robot <gen dir>,... --human take_003,... --out <dir> [--K 24] [--hop 48]
  python wam/encode_va.py --human-manifest Gitignore-but-important/wam/human_clean.json --out <dir>   (whole takes)

Writes <out>/{robot,human}/<source>_<start>.pt: latent (48, 1+K, 10, 18) bf16, actions
(1+K, 8, 8) float32 raw (robot only; row 0 unused), source, start; and <out>/text.pt (T5
embeddings of wam/_lingbot.PROMPTS).
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
from wam.render_va import H, H_RENDER, W  # noqa: E402

CAPTURES = os.path.join(ROOT, "Gitignore-but-important/laptop/microwave01")


def human_frames(take, t0=0.0, t1=None):
    """10 fps frames (every 3rd of the 30 fps take) from t0 to t1 seconds"""
    cap = cv2.VideoCapture(os.path.join(CAPTURES, take + ".avi"))
    out, k, k0 = [], 0, int(round(t0 * 30))
    k1 = int(round(t1 * 30)) if t1 is not None else None
    while True:
        ok, bgr = cap.read()
        if not ok or (k1 is not None and k >= k1):
            break
        if k >= k0 and (k - k0) % 3 == 0:
            out.append(cv2.resize(bgr, (W, H_RENDER), interpolation=cv2.INTER_AREA)[1:1 + H, :, ::-1])
        k += 1
    return np.stack(out)


def robot_actions(run_dir):
    z = os.path.join(run_dir, "demo.npz")
    if os.path.exists(z):
        return np.load(z)["actions"].astype(np.float32)
    import h5py
    with h5py.File(os.path.join(run_dir, "solve.h5"), "r") as f:
        return f["traj_0/actions"][:].astype(np.float32)


class Encoder:
    def __init__(self, dev):
        from modules.utils import WanVAEStreamingWrapper, load_vae
        self.vae = load_vae(os.path.join(LB.base_dir(), "vae"), torch.bfloat16, dev)
        self.s = WanVAEStreamingWrapper(self.vae)
        self.dev = dev
        self.mean = torch.tensor(self.vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
        self.inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, frames):
        """(1 + 4K, H, W, 3) uint8 -> (48, 1 + K, H/16, W/16) bf16"""
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.dev).permute(3, 0, 1, 2)[None].float()
        x = (x / 255.0 * 2.0 - 1.0).to(torch.bfloat16)
        self.s.clear_cache()
        outs = [self.s.encode_chunk(x[:, :, :1])]
        for t in range(1, x.shape[2], 16):
            outs.append(self.s.encode_chunk(x[:, :, t:t + 16]))
        mu = torch.chunk(torch.cat(outs, 2), 2, dim=1)[0]
        return ((mu.float() - self.mean) * self.inv_std)[0].to(torch.bfloat16).cpu()


@torch.no_grad()
def text_embeddings(dev="cpu"):
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from modules.utils import load_text_encoder, load_tokenizer
    tok = load_tokenizer(os.path.join(LB.base_dir(), "tokenizer"))
    te = load_text_encoder(os.path.join(LB.base_dir(), "text_encoder"), torch.bfloat16, dev)
    out = {}
    for k, p in LB.PROMPTS.items():
        ti = tok([prompt_clean(p)], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                 return_attention_mask=True, return_tensors="pt")
        e = te(ti.input_ids.to(dev), ti.attention_mask.to(dev)).last_hidden_state
        n = int(ti.attention_mask.sum())
        e = torch.cat([e[0, :n], e.new_zeros(512 - n, e.shape[-1])])[None]   # as the server: zeros past the prompt
        out[k] = e.to(torch.bfloat16).cpu()
    return out


def clips(n, K, hop):
    L = 1 + 4 * K
    return list(range(0, n - L + 1, hop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="", help="comma-separated generator dirs (runs with va_front_splat.npz)")
    ap.add_argument("--human", default="", help="comma-separated takes")
    ap.add_argument("--human-manifest", help="human_clean.json: one clip per kept take, its whole [start, end] span "
                                             "(cut to 1 + 4 K frames), with the take's prompt key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--hop", type=int, default=48)
    ap.add_argument("--video", default="va_front_splat")
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--max-runs", type=int, default=0, help="robot: first N runs per generator dir (smoke test)")
    ap.add_argument("--whole", action="store_true", help="robot: one clip per run, the whole demonstration "
                                                          "(cut to 1 + 4 K frames) instead of K-frame clips every --hop")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.out, "text.pt")):
        os.makedirs(a.out, exist_ok=True)
        torch.save(text_embeddings(a.text_device), os.path.join(a.out, "text.pt"))
        print("text embeddings", flush=True)
    enc = Encoder(torch.device("cuda"))
    L = 1 + 4 * a.K
    n_clips = {"robot": 0, "human": 0}
    for d in [x for x in a.robot.split(",") if x]:
        s = json.load(open(os.path.join(d, "summary.json")))
        for r in [r for r in s["runs"] if r["success"]][:a.max_runs or None]:
            rd = os.path.join(d, f"run_{r['run']:03d}")
            v = os.path.join(rd, a.video + ".npz")
            if not os.path.exists(v):
                print("  missing", v, flush=True)
                continue
            z = np.load(v)
            front, idx = z["front"], z["idx"]
            acts = robot_actions(rd)
            src = f"{os.path.basename(d)}_{r['run']:03d}"
            os.makedirs(os.path.join(a.out, "robot"), exist_ok=True)
            K = (len(front) - 1) // 4 if a.whole else a.K
            Lr = 1 + 4 * K
            for s0 in ([0] if a.whole else clips(len(front), a.K, a.hop)):
                f = idx[s0:s0 + Lr]
                act = np.zeros((1 + K, 8, 8), np.float32)
                for j in range(1, 1 + K):
                    seg = acts[f[4 * (j - 1)]:f[4 * j]]
                    assert len(seg) == 8, (src, s0, j, len(seg))
                    act[j] = seg
                torch.save(dict(latent=enc(front[s0:s0 + Lr]), actions=torch.from_numpy(act), source=src, start=int(s0),
                                camera=str(z["take"])), os.path.join(a.out, "robot", f"{src}_{s0:05d}.pt"))
                n_clips["robot"] += 1
            print(f"  {src}: {len(front)} frames, {n_clips['robot']} robot clips so far", flush=True)
    for t in [x for x in a.human.split(",") if x]:
        fr = human_frames(t)
        os.makedirs(os.path.join(a.out, "human"), exist_ok=True)
        for s0 in clips(len(fr), a.K, a.hop):
            torch.save(dict(latent=enc(fr[s0:s0 + L]), actions=None, source=t, start=int(s0), camera=t),
                       os.path.join(a.out, "human", f"{t}_{s0:05d}.pt"))
            n_clips["human"] += 1
        print(f"  {t}: {len(fr)} frames, {n_clips['human']} human clips so far", flush=True)
    if a.human_manifest:
        os.makedirs(os.path.join(a.out, "human"), exist_ok=True)
        for t, k in json.load(open(a.human_manifest))["keep"].items():
            fr = human_frames(t, k["start"], k["end"])
            fr = fr[:1 + 4 * ((len(fr) - 1) // 4)]
            torch.save(dict(latent=enc(fr), actions=None, source=t, start=int(round(k["start"] * 10)), camera=t,
                            prompt=k["prompt"]), os.path.join(a.out, "human", f"{t}_full.pt"))
            n_clips["human"] += 1
            print(f"  {t}: {len(fr)} frames ({k['start']}-{k['end']} s), prompt {k['prompt']}", flush=True)
    print(n_clips)


if __name__ == "__main__":
    main()
