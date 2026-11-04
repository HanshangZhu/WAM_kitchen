# LingBot-VA post-training: next steps (4 November 2026)

These are the experiments still to run after `EXPERIMENTS.md`, each with what it answers and what it costs. Section
numbers refer to `EXPERIMENTS.md`, and literature notes are in `literature/`.

## Measured unit costs

- **Training, RTX 5090:** 10.5 s per step, i.e. 2.9 GPU-hours per 1000 steps; 20000 steps from base take about 58 h.
- **Closed loop, 24 starts queried every 32 steps:** about 15 min on the RTX 5090 (48 starts: 26 min). The DGX Spark
  takes about 50 min (estimated).

Evaluation is cheap and training is the bottleneck. So evaluation-only experiments come first, ordered below by
information per GPU-hour.

## A. Evaluation only: about 4 GPU-hours in total

| experiment | cost | answers |
|---|---|---|
| **Interpolate toward the base:** scale the final model's LoRA delta by λ ∈ {0, 0.25, 0.5, 0.75, 1}; validation losses at all five, closed loop at λ = 1, 0.75, 0.5 on starts 8..31 | ~1 h (RTX 5090) | Does some λ keep the success while the video prior drifts less? This needs no replay data (literature/forgetting.md: WiSE-FT, RETAIN). λ = 0 is "base + our action interface", not the raw base |
| **The final model on the second machine,** starts 8..31 | ~50 min (DGX Spark) | Cross-machine check of 71/72 (section 13). It comes free as λ = 1 if the sweep runs there |
| **An out-of-distribution probe:** 2 scene perturbations (layout, lighting or object pose) × {final model, best λ, 12000 steps} × 24 starts | ~1.5 h, plus coding the perturbations | Does keeping the video prior buy generalisation? Without it, the human-video loss has no shown value (section 18) |
| **Clean vs own history:** closed loop fed the matching demonstration's clean action history | ~30 min, plus code | Separates copycat from exposure bias, the alternative section 15 leaves open |
| **Re-run the hand-over starts 27 and 31,** logging the final pot pose and door angle | < 30 min | Stage labels for the rescue (section 16) |
| **A held-out video set no run trains on** | validation passes only | A clean forgetting measure. The ego takes are trained on in the co-training run (section 18) |
| **Velocity gap to the LoRA-off model** on held-out human clips, over the saved checkpoints, and the human-loss rise per noise level | forward passes only | Where in noise level the drift sits, and what a distillation term would act on (literature/kl_to_base.md) |

## B. Training: about 800 GPU-hours if everything ran

| experiment | cost | answers |
|---|---|---|
| **Recovery test:** 200-500 steps on human video only | ~1-1.5 h | Spurious (alignment) vs real forgetting: does the loss return to about 0.25 quickly? |
| **Continue 14000 -> 20000 at constant lr,** with the same optimizer restart | ~17.5 h | Splits the 76 -> 99 % gain between more steps and the lr decay, with the restart held fixed |
| **A second training seed of the final recipe** (noise, 14000 constant + 6000 with decay) | ~58 h | Is 71/72 a property of the recipe or of one run? |
| **A second seed of the no-noise control** (14000) | ~41 h | Is 16 vs 5 larger than retraining variance? |
| **Noise dose:** p ∈ {0.25, 0.75, 1 = drop the action history}, 8000 steps each | ~69 h | Is p = 0.5 near the best? An inverted U (as in AR-VLA) would mean the history is useful but must be de-weighted |
| **An equal-strength regulariser that leaves the action history alone** | ~41 h | Copycat remedy vs generic regularisation |
| **Recipe components removed one at a time:** retry demos, mid-task starts, pretrain action layout, full-window attention (14000 steps each) | ~164 h | Which parts are needed |
| **Retry without planned misses:** mask the actions *and* the video targets of the scripted miss, or replace scripted misses by external, visible failures | 41-58 h, plus CPU data generation | Teaching retry without teaching the miss (section 17; literature/recovery_without_imitating_failures.md) |
| **Replay matched on 20000 robot updates:** 25 % ego video first (the strong-shift reference value), later 5 / 15 / 30 % | ~78 h for 25 %; ~213 h for the other three | Replay's cost and benefit without the step confound of section 11 |
| **Distillation to the LoRA-off model on human clips only,** video velocity only, β ∈ {0.3, 1, 3}, alone and with replay | about the replay run's cost + ~10 % | Does an anchor to the base work without replay, or with it? It is a W2-type surrogate, not a KL (literature/kl_to_base.md) |

## C. Further ablations

These follow from the findings in `EXPERIMENTS.md`. Training runs go from base for 14000 steps and are read on all 72
starts against the noise model at the same step.

| experiment | cost | answers |
|---|---|---|
| **Context window at inference:** attend to the last K chunks only, K ∈ {1, 4, 16, all}; on starts 8..31 and the 17 hand-overs that inherit the history | ~2 h, plus code | How much of the 71/72 needs the whole episode? Does a short window let a model recover when it inherits a failed history (0/17 now; section 16)? |
| **Reset the cache when a grasp fails** (the pot is < 5 cm up 32 steps after the gripper closed), all 72 starts, no-noise model and final model | ~1.5 h, plus code | The deployable form of the empty-cache rescue: do the 10/17 re-grasps turn into a higher success rate? |
| **Classifier-free guidance** ∈ {1, 3, 5 (used), 8}, final model, starts 8..31 | ~1 h | Does guidance matter for a single-task policy? Guidance 1 also halves the compute per query |
| **Random history truncation in training** (each sample's context starts at a random chunk), with the action-history noise | ~41 h, plus code | Recovery from an inherited failed history, and fewer planned door misses (cutting the failure prefix, as in the recovery literature) |
| **Without the handle-miss retry demos** | ~41 h | Do the planned door misses disappear, and what does it cost? |
| **Final recipe + wrist camera** (the wrist was only tried without the noise) | ~50 h, plus rendering | Does the wrist view help once the policy looks at the video? |
| **LoRA rank 32 instead of 128** | ~41 h | Task vs forgetting: lower rank is reported to forget less (`literature/forgetting.md`) |
| **The action-history noise without LingBot-VA's video-history noise** | ~41 h, plus a switch | Is the recipe's own video noise needed too? |
| **Demo-count curve:** 25 % and 50 % of the data, same mix | ~82 h | Sample efficiency |

The three inference-only rows belong in the first block of the order below. The training rows compete for the
multi-day slot.

## Order

1. **The evaluation-only block (A),** plus the recovery test: about 5 h.
2. **The constant-lr continuation:** 17.5 h.
3. **Then one multi-day slot,** either the second seed of the final recipe or the matched replay / distillation pair.
   - The second seed tests the recipe claim.
   - The replay / distillation pair tests whether the task can be gained without the drift.
