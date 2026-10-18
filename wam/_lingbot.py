"""Import LingBot-VA's own code (github.com/Robbyant/lingbot-va, checked out at LINGBOT_VA_REPO) on
one GPU without flash-attn: wan_va imports flash_attn_interface at module level but only uses it
for attn_mode="flashattn"; we use "torch" (inference) and "flex" (training)."""
import os
import sys
import types

REPO = os.environ.get("LINGBOT_VA_REPO", os.path.expanduser("~/Research/WAM/WAM-OPD/lingbot-va"))

from diffusers import AutoencoderKLWan  # noqa: F401,E402  (resolve diffusers' lazy imports before the stub)
from diffusers.models.attention import FeedForward  # noqa: F401,E402

if "flash_attn_interface" not in sys.modules:
    _fa = types.ModuleType("flash_attn_interface")
    _fa.flash_attn_func = None
    sys.modules["flash_attn_interface"] = _fa
for p in (os.path.join(REPO, "wan_va"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE = os.environ.get("LINGBOT_VA_BASE", "robbyant/lingbot-va-base")

PROMPTS = {
    "robot": "a robot arm opens the microwave, puts the pot inside the microwave, and closes the microwave door",
    "human": "a person opens the microwave, puts the pot inside the microwave, and closes the microwave door",
    "human_adjust": "a person adjusts the pot position before opening the microwave door, then puts the pot inside the microwave, and closes the microwave door",
    "empty": "",
}


def base_dir():
    """local snapshot of the base checkpoint"""
    if os.path.isdir(BASE):
        return BASE
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE)
