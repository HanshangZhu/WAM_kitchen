"""Average several wam/train_va.py checkpoints of one run into one checkpoint that eval_va.py loads unchanged.

Each checkpoint's model is W0 + B_i A_i for every LoRA linear (alpha = rank, so the scale is 1) plus its own action
layers. The average of the k models is W0 + (1/k) sum_i B_i A_i, which is stored exactly as one LoRA of rank k*r:
A = [A_1; ...; A_k] (stacked rows) and B = [B_1, ..., B_k] / k (stacked columns), so that B A = (1/k) sum_i B_i A_i.
eval_va.merge_lora adds B A for whatever rank it finds. The action layers (full tensors) are averaged directly.
The run's meta.json is copied next to the output with the list of averaged checkpoints added.

    python wam/average_lora.py --out <dir>/lora_018000.pt <run>/lora_014000.pt ... <run2>/lora_018000.pt
"""
import argparse
import json
import os

import torch


def average(paths):
    sds = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = list(sds[0])
    assert all(list(sd) == keys for sd in sds), "the checkpoints have different keys"
    k, out = len(sds), {}
    for key in keys:
        if key.endswith(".A"):
            out[key] = torch.cat([sd[key] for sd in sds], 0)
        elif key.endswith(".B"):
            out[key] = torch.cat([sd[key] for sd in sds], 1) / k
        else:
            out[key] = sum(sd[key] for sd in sds) / k
    return sds, out


def check(sds, out, n=5):
    """max |B A - mean_i B_i A_i| relative to max |mean_i B_i A_i| over the first n LoRA linears"""
    worst = 0.0
    for key in [x for x in out if x.endswith(".A")][:n]:
        b = key[:-2] + ".B"
        want = sum(sd[b].double() @ sd[key].double() for sd in sds) / len(sds)
        got = out[b].double() @ out[key].double()
        worst = max(worst, float((got - want).abs().max() / want.abs().max()))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="lora_*.pt files, all from runs with the same settings")
    ap.add_argument("--out", required=True, help="output lora_*.pt; its folder gets the meta.json")
    a = ap.parse_args()
    sds, out = average(a.ckpts)
    print(f"averaged {len(sds)} checkpoints; exactness check: relative error {check(sds, out):.2e}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    meta = json.load(open(os.path.join(os.path.dirname(a.ckpts[-1]), "meta.json")))
    meta["averaged_from"] = a.ckpts
    json.dump(meta, open(os.path.join(os.path.dirname(a.out), "meta.json"), "w"), indent=1)
    print("wrote", a.out, flush=True)
