# Literature notes

The reading behind the main decisions in `../EXPERIMENTS.md` and `../NEXT_STEPS.md`. Each note ends with what applies
to this setup and with experiments, ordered by compute.

| note | question |
|---|---|
| `action_history_noise.md` | Why does noise on the action history matter so much, and who has found the same? |
| `recovery_without_imitating_failures.md` | How do others teach retrying without the policy learning to fail on purpose? |
| `forgetting.md` | How can the task be learned without forgetting the video prior (2025-26 consensus)? |
| `kl_to_base.md` | What does "KL to the base" mean for diffusion and flow models, and does a distillation term toward the base help? |

**How to read the citations.** Citations give the venue and the arXiv id, and numbers were read in the papers' text.
The markers say where that was not possible:

| marker | meaning |
|---|---|
| UNVERIFIED | The venue could not be confirmed, or only the metadata or abstract was read. |
| [ext] | The number came from a summarised fetch of the paper, not a direct reading; spot-check before quoting. |
| fig-only | The result appears only in a figure; no number was read. |
| layout reconstructed | The table was re-assembled by hand from extracted text; spot-check before quoting. |
| "arXiv" in place of a venue | No venue was found. |
