#!/usr/bin/env python
"""LoRA post-training of LingBot-VA base on one GPU: robot clips (video + actions) and,
optionally, human ego clips (video only, actions masked out of the loss and the input).

Noising, the teacher-forced causal-chunk input and the frame-wise losses are LingBot-VA's own
(wan_va/train.py: Trainer._add_noise / _prepare_input_dict / compute_loss, copied below so that
wandb / FSDP / LeRobot are not needed); the transformer is the repo's WanTransformer3DModel with
flex attention and activation checkpointing (wan_va/distributed/fsdp.apply_ac).

Trained: rank-r LoRA on every linear layer of the 30 blocks (self-attention, cross-attention,
FFN), plus the action embedder / action output projection in full (the 8 used action channels
mean something new: pd_joint_pos, 7 joints + gripper, in slots 0..7 of the 30).

  python wam/train_va.py --data <encode_va out> --out <dir> [--human-frac 0.3] [--steps 3000]

Robot runs listed in --val-robot and takes in --val-human are held out; their frame-wise video
loss at fixed noise is logged every --val-every optimiser steps (the metric for the ego question).
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wam import _lingbot as LB  # noqa: E402
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, sample_timestep_id  # noqa: E402

N_ACT = 8                       # used action channels (slots 0..7 of 30)
ACTION_DIM, ACT_PER_FRAME = 30, 8
SNR_SHIFT, ACTION_SNR_SHIFT = 5.0, 1.0      # va_demo_cfg (the base checkpoint's demo config)
PATCH = (1, 2, 2)


# ---- data ----------------------------------------------------------------------------------
class Clips(torch.utils.data.Dataset):
    def __init__(self, files, text, q01, q99, cfg_prob=0.1):
        self.files, self.text, self.q01, self.q99, self.cfg_prob = files, text, q01, q99, cfg_prob

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], weights_only=False)
        lat = d["latent"]                                            # (48, F, h, w)
        Fn = lat.shape[1]
        act = torch.zeros(ACTION_DIM, Fn, ACT_PER_FRAME, 1)
        mask = torch.zeros(ACTION_DIM, Fn, ACT_PER_FRAME, 1, dtype=torch.bool)
        if d["actions"] is not None:
            a = d["actions"].float()                                 # (F, 8 steps, 8 ch) raw
            a = ((a - self.q01) / (self.q99 - self.q01 + 1e-6) * 2 - 1).clamp(-1.5, 1.5)
            act[:N_ACT, 1:] = rearrange(a[1:], "f n c -> c f n 1")
            mask[:N_ACT, 1:] = True                                  # frame 0 (the first image) has no actions
        kind = "robot" if d["actions"] is not None else "human"
        txt = self.text["empty" if random.random() < self.cfg_prob else d.get("prompt", kind)][0]
        return dict(latents=lat.float(), actions=act * mask, actions_mask=mask, text_emb=txt.float())


def action_stats(files):
    a = torch.cat([torch.load(f, weights_only=False)["actions"][1:].reshape(-1, N_ACT) for f in files])
    q01, q99 = torch.quantile(a, 0.01, dim=0), torch.quantile(a, 0.99, dim=0)
    gripper = N_ACT - 1                       # +1 open / -1 close: keep the full range
    q01[gripper], q99[gripper] = -1.0, 1.0
    return q01, q99


# ---- LoRA ----------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B) * self.s).to(x.dtype)


def add_lora(model, r, alpha):
    n = 0
    for blk in model.blocks:
        for parent in (blk.attn1, blk.attn2):
            for name in ("to_q", "to_k", "to_v"):
                setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
                n += 1
            parent.to_out[0] = LoRALinear(parent.to_out[0], r, alpha)
            n += 1
        blk.ffn.net[0].proj = LoRALinear(blk.ffn.net[0].proj, r, alpha)
        blk.ffn.net[2] = LoRALinear(blk.ffn.net[2], r, alpha)
        n += 2
    return n


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B") or k.startswith(("action_embedder", "action_proj_out"))}


# ---- LingBot-VA's training step (wan_va/train.py Trainer, unchanged logic) -----------------
class Step:
    def __init__(self, dev, full_window_p=0.0):
        self.device = dev
        self.full_window_p = full_window_p          # P(attention over the whole sample) instead of the 4..64 window
        self.patch_size = PATCH
        self.gradient_accumulation_steps = 1
        self.train_scheduler_latent = FlowMatchScheduler(shift=SNR_SHIFT, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=ACTION_SNR_SHIFT, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F_, H, W = latent.shape
        timestep_ids = sample_timestep_id(batch_size=F_, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)
        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        latent_grid_id = get_mesh_id(latent.shape[-3] // patch_f, latent.shape[-2] // patch_h,
                                     latent.shape[-1] // patch_w, t=1 if action_mode else 0, f_w=1, f_shift=0,
                                     action=action_mode).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)
        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(batch_size=F_, min_timestep_bd=0.5, max_timestep_bd=1.0,
                                                   num_train_timesteps=train_scheduler.num_train_timesteps)
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)
        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()
        return dict(timesteps=timesteps[None].repeat(B, 1), noisy_latents=noisy_latents, targets=targets,
                    latent=latent, cond_timesteps=cond_timesteps[None].repeat(B, 1), grid_id=latent_grid_id)

    @torch.no_grad()
    def prepare(self, b, chunk_size=None, window_size=None):
        latent_dict = self._add_noise(b["latents"], self.train_scheduler_latent, action_mask=None, action_mode=False,
                                      noisy_cond_prob=0.5)
        action_dict = self._add_noise(b["actions"], self.train_scheduler_action, action_mask=b["actions_mask"],
                                      action_mode=True, noisy_cond_prob=0.0)
        latent_dict["text_emb"] = b["text_emb"]
        action_dict["text_emb"] = b["text_emb"]
        action_dict["actions_mask"] = b["actions_mask"]
        return {"latent_dict": latent_dict, "action_dict": action_dict,
                "chunk_size": chunk_size or torch.randint(1, 5, (1,)).item(),
                "window_size": window_size or (100000 if random.random() < self.full_window_p
                                               else torch.randint(4, 65, (1,)).item())}

    def loss(self, input_dict, pred):
        """(video loss, action loss), frame-wise as wan_va/train.py compute_loss (no accumulation scaling)"""
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=input_dict["action_dict"]["targets"].shape[-3])
        t = input_dict["latent_dict"]["targets"]
        latent_pred = data_seq_to_patch(self.patch_size, latent_pred, t.shape[-3], t.shape[-2], t.shape[-1],
                                        batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
        lw = self.train_scheduler_latent.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(Bn, Fn)
        aw = self.train_scheduler_action.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(Bn, Fn)
        ll = F.mse_loss(latent_pred.float(), t.float(), reduction="none") * lw[:, None, :, None, None]
        ll = ll.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).mean(1).mean()
        al = F.mse_loss(action_pred.float(), input_dict["action_dict"]["targets"].float(), reduction="none")
        m = input_dict["action_dict"]["actions_mask"].float()
        al = al * aw[:, None, :, None, None] * m
        al = al.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(1)
        mm = m.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(1)
        al = (al / (mm + 1e-6)).mean()
        return ll, al


# ---- main ----------------------------------------------------------------------------------
def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def val_loss(model, step, files, ds, dev, n=24):
    """mean frame-wise video (and action) loss over up to n held-out clips at fixed noise / timesteps"""
    if not files:
        return float("nan"), float("nan")
    model.eval()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
    vl, al = [], []
    for k, f in enumerate(files[:n]):
        torch.manual_seed(1000 + k)
        random.seed(1000 + k)
        b = {kk: v[None] for kk, v in Clips([f], ds.text, ds.q01, ds.q99, 0.0)[0].items()}
        inp = step.prepare(to_dev(b, dev), chunk_size=2, window_size=100000 if step.full_window_p > 0 else 32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ll, aa = step.loss(inp, model(inp, train_mode=True))
        vl.append(ll.item())
        if b["actions_mask"].any():
            al.append(aa.item())
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    random.setstate(rng[2])
    model.train()
    return float(np.mean(vl)), float(np.mean(al)) if al else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="wam/encode_va.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--human-frac", type=float, default=0.0, help="fraction of samples drawn from the human clips")
    ap.add_argument("--full-window-p", type=float, default=0.0,
                    help="probability of attending over the whole sample (whole-episode training); "
                         "otherwise LingBot-VA's random 4..64 window")
    ap.add_argument("--robot-runs", type=int, default=0, help="use only the first N training robot runs (0: all)")
    ap.add_argument("--val-robot", default="gen_011,gen_023,gen_035,gen_047,gen_dart02_011,gen_dart02_041")
    ap.add_argument("--val-human", default="take_027,take_035,take_041,take_045,take_050,take_053")
    ap.add_argument("--steps", type=int, default=3000, help="optimiser steps")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="continue from a lora_*.pt (weights only; AdamW restarts, with the 50-step warmup)")
    ap.add_argument("--start-step", type=int, default=0, help="step number of --init (logging / checkpoint names)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed + a.start_step)
    torch.manual_seed(a.seed + a.start_step)
    dev = torch.device("cuda")

    val_r = tuple(a.val_robot.split(","))
    val_h = tuple(a.val_human.split(","))
    robot = sorted(glob.glob(os.path.join(a.data, "robot", "*.pt")))
    human = sorted(glob.glob(os.path.join(a.data, "human", "*.pt")))
    src = lambda f: os.path.basename(f).rsplit("_", 1)[0]                      # noqa: E731
    r_val = [f for f in robot if src(f) in val_r]
    r_tr = [f for f in robot if src(f) not in val_r]
    if a.robot_runs:
        keep = sorted({src(f) for f in r_tr})[:a.robot_runs]
        r_tr = [f for f in r_tr if src(f) in keep]
    h_val = [f for f in human if src(f) in val_h]
    h_tr = [f for f in human if src(f) not in val_h]
    random.Random(0).shuffle(r_val)
    random.Random(0).shuffle(h_val)
    q01, q99 = action_stats(r_tr)
    text = torch.load(os.path.join(a.data, "text.pt"))
    ds_r = Clips(r_tr, text, q01, q99)
    ds_h = Clips(h_tr, text, q01, q99)
    meta = dict(vars(a), n_robot_train=len(r_tr), n_robot_runs=len({src(f) for f in r_tr}), n_human_train=len(h_tr),
                n_robot_val=len(r_val), n_human_val=len(h_val), q01=q01.tolist(), q99=q99.tolist(),
                prompts=LB.PROMPTS, action="pd_joint_pos 7 joints + gripper (+1 open / -1 close), slots 0..7")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("q01", "q99", "prompts")}), flush=True)

    from modules.utils import load_transformer
    from distributed.fsdp import apply_ac
    model = load_transformer(os.path.join(LB.base_dir(), "transformer"), torch_dtype=torch.bfloat16,
                             torch_device="cpu", attn_mode="flex")
    model.requires_grad_(False)
    n_lora = add_lora(model, a.rank, alpha=a.rank)
    for m in (model.action_embedder, model.action_proj_out):
        m.float().requires_grad_(True)
    apply_ac(model)
    if a.init:
        sd = torch.load(a.init, weights_only=False)
        own = model.state_dict()
        for k, v in sd.items():                 # a lower-rank init goes into the first rows of A / columns of B;
            if k in own and v.shape != own[k].shape:    # the extra B columns stay zero, so the function is unchanged
                pad = own[k].clone()
                if k.endswith(".A"):
                    pad[:v.shape[0]] = v
                else:
                    pad[:, :v.shape[1]] = v
                sd[k] = pad
        bad = model.load_state_dict(sd, strict=False).unexpected_keys
        assert not bad, bad[:5]
        print(f"init from {a.init}: {len(sd)} tensors", flush=True)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA on {n_lora} linears, {sum(p.numel() for p in params) / 1e6:.1f} M trainable", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    step = Step(dev, a.full_window_p)
    model.train()

    def sample():
        ds = ds_h if (ds_h and random.random() < a.human_frac) else ds_r
        return {k: v[None] for k, v in ds[random.randrange(len(ds))].items()}

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for s in range(a.start_step, a.steps + 1):
        if s % a.val_every == 0:
            vr, ar = val_loss(model, step, r_val, ds_r, dev)
            vh, _ = val_loss(model, step, h_val, ds_r, dev)
            rec = dict(step=s, val_video_robot=vr, val_action_robot=ar, val_video_human=vh, t=round(time.time() - t0))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if s % a.save_every == 0 and s > a.start_step or s == a.steps:
            torch.save(trainable_state(model), os.path.join(a.out, f"lora_{s:06d}.pt"))
        if s == a.steps:
            break
        lv, la = [], []
        for _ in range(a.accum):
            inp = step.prepare(to_dev(sample(), dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ll, al = step.loss(inp, model(inp, train_mode=True))
            ((ll + al) / a.accum).backward()
            lv.append(ll.item())
            la.append(al.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0:
            rec = dict(step=s, video=float(np.mean(lv)), action=float(np.mean(la)), grad=float(gn),
                       s_per_step=round((time.time() - t0) / (s - a.start_step + 1), 2), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
