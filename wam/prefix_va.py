#!/usr/bin/env python
"""Imagination after a real prefix: the first P seconds of a held-out solver demo (its front video and
executed actions) go into the KV cache chunk by chunk, exactly as the closed loop feeds what it saw
(wam/eval_va.py), then the model imagines on from there with nothing executed. If it imagines the
door opening once the arm is already at the handle, the post-trained model knows the event and the
closed-loop failure is in getting there; if not, it has not learnt the event.

  python wam/prefix_va.py --lora <run>/lora_005000.pt --run Gitignore-but-important/sim/gen/run_011 \
      --prefix 8,13,17 --out <dir>

Writes <out>/p<P>_s<seed>.mp4 (top: the demo, bottom: the real prefix then the imagination) and
<out>/sheet.png (per row: the last prefix frame, then the demo and the imagination every 2 s).
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
from wam.encode_va import robot_actions  # noqa: E402
from wam.eval_va import CAM_KEY, make_config, merge_lora  # noqa: E402

FIRST, CHUNK = 12, 16      # video frames (10 fps) the closed loop adds per chunk: 3 latent frames, then 4


def chunk_actions(acts, idx, c):
    """(8 dims, 4 latent frames, 8 steps) for chunk c, as wam/encode_va.py pairs them (latent 0: zeros)"""
    out = np.zeros((8, 4, 8), np.float32)
    for f in range(4):
        j = 4 * c + f
        if j:
            out[:, f, :] = acts[idx[4 * (j - 1)]:idx[4 * j]].T
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--run", required=True, help="a solver run dir with va_front_splat.npz (a held-out one)")
    ap.add_argument("--prefix", default="8,13,17", help="seconds of real demo fed before imagining")
    ap.add_argument("--chunks", type=int, default=10, help="imagined chunks after the prefix (1.6 s each)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--prompt", default="robot")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(os.path.dirname(a.lora), "meta.json")))
    z = np.load(os.path.join(a.run, "va_front_splat.npz"))
    front, idx = z["front"], z["idx"]
    acts = robot_actions(a.run)

    import imageio
    import wan_va.wan_va_server as S
    from diffusers.video_processor import VideoProcessor
    from modules.utils import WanVAEStreamingWrapper
    S.init_logger()
    m = S.VA_Server(make_config(meta, a.guidance, os.path.join(a.out, "_server")))
    m.vae = m.vae.to(m.device)
    m.streaming_vae = WanVAEStreamingWrapper(m.vae)
    m.video_processor = VideoProcessor(vae_scale_factor=1)
    print(f"merged LoRA into {merge_lora(m.transformer, torch.load(a.lora, weights_only=False))} linears", flush=True)

    m._reset(LB.PROMPTS.get(a.prompt, a.prompt))         # T5 runs on the CPU: encode the prompt once
    emb = m.prompt_embeds, m.negative_prompt_embeds
    rows = []
    for P in [float(x) for x in a.prefix.split(",")]:
        n_pre = 1 + int(np.ceil(max(0.0, P * 10 - FIRST) / CHUNK))
        end = FIRST + CHUNK * (n_pre - 1)              # last real frame index in the cache
        for s in range(a.seeds):
            torch.manual_seed(s)
            m._reset(None)
            m.prompt_embeds, m.negative_prompt_embeds = emb
            for c in range(n_pre):
                lo = 0 if c == 0 else FIRST + CHUNK * (c - 1)
                m._infer({"obs": [{CAM_KEY: front[lo]}]}, frame_st_id=m.frame_st_id)   # as the loop; discarded
                keys = [{CAM_KEY: front[k]} for k in range(lo + 1, lo + 1 + (FIRST if c == 0 else CHUNK))]
                m._compute_kv_cache({"obs": keys, "state": chunk_actions(acts, idx, c)})
            lat = []
            for k in range(a.chunks):
                _, l = m._infer({"obs": [{CAM_KEY: front[end]}]}, frame_st_id=m.frame_st_id + 4 * k)
                lat.append(l)
            v = m.decode_one_video(torch.cat(lat, 2), "np")[0]
            v = (np.clip(v, 0, 1) * 255).astype(np.uint8) if v.dtype != np.uint8 else v
            # the first imagined latent covers frames end+1..end+4 but decodes to one frame (~end+4)
            gt = front[end + 4:end + 4 + len(v)]
            n = min(len(v), len(gt))
            mine = np.concatenate([front[:end + 1], np.repeat(v[:1], 3, 0), v[:n]])
            demo = front[:end + 4 + n]
            vid = [np.vstack([demo[k], mine[k]]) for k in range(len(mine))]
            for k in range(end + 1, len(vid)):
                cv2.putText(vid[k], "imagined", (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            imageio.mimwrite(os.path.join(a.out, f"p{P:g}_s{s}.mp4"), vid, fps=10, codec="libx264", quality=7,
                             macro_block_size=1)
            t = np.arange(0, n, 20)[:7]
            top = np.hstack([front[end]] + [gt[i] for i in t])
            bot = np.hstack([front[end]] + [v[i] for i in t])
            row = np.ascontiguousarray(np.vstack([top, bot]), dtype=np.uint8).copy()
            cv2.putText(row, f"prefix {P:g}s (to {end / 10:.1f}s) s{s}: top demo, bottom imagined; +2s/col",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            rows.append(row)
            print(f"prefix {P}s -> cache to frame {end} ({end / 10:.1f}s), imagined {len(v)} frames", flush=True)
    cv2.imwrite(os.path.join(a.out, "sheet.png"), np.vstack(rows)[..., ::-1])


if __name__ == "__main__":
    main()
