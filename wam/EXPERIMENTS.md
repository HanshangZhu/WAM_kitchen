# LingBot-VA post-training: log (last update 2026-11-04)

Everything about post-training LingBot-VA on the twin's solver demonstrations, 18 October to 4 November 2026: one
section per question, roughly in the order they were asked. The README's "Results (step 2a)" is the short version;
this file holds the runs behind it, the numbers, the tests and what did not work. Runs are named by what changed.
Checkpoints, rollouts and videos are not in the repository (videos are available on request).

## Summary

| model (each named by what it changed) | starts | success |
|---|---|---|
| zero-shot | -- | 0/21 |
| first post-trained models (joints layout, 18-19 Oct) | starts 0..7 | 0-1 of 8 |
| combined demos, pretrain action layout, IK execution, continued to 14000 steps (DGX Spark) | 8..31 | 8/24 |
| slow demos + wrist camera + retry demos, 20000 steps (DGX Spark) | 8..31 | 19/24 |
| the same model (RTX 5090) | 8..79 | 43/72 (60 %) |
| combined demos + handle-miss retries + **action-history noise**, 12000 steps (RTX 5090) | 8..79 | 55/72 (76 %) |
| the same data without the noise, 8000 steps (best of 7 checkpoints on starts 0..7) | 8..31 | 5/24 |
| the noise model + 30 % ego human video, 10000 steps (best of 7) | 8..31 | 11/24 |
| **the noise model continued 14000 -> 20000 with lr decay** (RTX 5090) | 8..79 | **71/72 (99 %, 95 % CI 93-100)** |

## Setup

**Model.** `lingbot-va-base`, the open-source LingBot-VA (arXiv 2601.21998) on Wan2.2.
- **Architecture:** a joint DiT. Each of the 30 blocks has one attn1 / attn2 / ffn shared by video and action tokens;
  this is not the paper's mixture-of-transformers.
- **Post-training (`train_va.py`):**
  - LoRA rank 128 on every attn1 / attn2 / ffn linear;
  - `action_embedder` and `action_proj_out` trained in full;
  - the action timestep embedding, scale-shift table and action norms frozen.
- **Loss:** video flow matching plus action loss, equal weight, while video tokens outnumber action tokens per frame 45-70
  to 8.
- **Optimiser:** AdamW (weight decay 0), lr 1e-4 unless stated, gradient clip 2.0.
- **Context:** attention over the whole episode in half of the training samples.

**Twin and task.** `MicrowavePot` from the first repository: a Franka Panda, the measured microwave, the Snap Pot and the
Gaussian-splat room behind the scene.
- The arm opens the door, puts the pot inside and shuts the door.
- **Success:** the pot is inside the cavity and flat on the turntable (|z − turntable top| < 0.02 m), and the door is shut
  (< 5°).
- **Stage counts** are also logged: door opened to ≥ 85°, and pot lifted > 10 cm.

**Observation and execution (`eval_va.py`).**
- **Camera:** the front camera at 160 × 288, from a held-out take's registered pose pulled back 0.3 m and up 0.1 m. When a
  run uses the wrist camera, it sits right of the front view in one 160 × 448 frame.
- **Queries:** the model is queried every 32 simulator steps (4 latent frames), or every 8 (1 latent frame) where stated.
- **Execution:** the predicted end-effector pose is executed by IK (`--exec ee_ik`).
- **Episode cap:** episodes stop at 1500 steps. Before 26 Oct the cap was 2600; over the 722 episodes run until then,
  every success ended by step 1229.

**Starts and statistics.**
- **Fixed starts:** the pot's starts come from a fixed list (`eval_va.pot_starts`), so every model meets the same start
  under the same number.
- **Starts 0..7** are used only to pick checkpoints. **Starts 8..79** never are: 8..31 first, then 32..79.
- **Picking rule,** fixed before the results: the most successes on starts 0..7, then the most doors opened to ≥ 85°,
  then the later step ("the rule's pick").
- **Tests:** comparisons are paired by start (exact McNemar), with Wilson 95 % intervals for rates.
- **Two machines:** evaluations ran on the DGX Spark (aarch64) until 2 Nov, a few early ones on a Jetson Thor, and on
  the RTX 5090 desktop (x86) from 3 Nov.

**Determinism and machines.**
- **One machine is deterministic.** The same weights on the same machine and container give the same episodes,
  byte-identical videos included: one checkpoint was re-run on starts 0..7 with the same five successes at the same steps.
  So re-running a checkpoint adds nothing; more starts do.
- **Machines differ (section 13).** Outcomes are sensitive to perturbations as small as the numerical differences
  between machines, so **results are compared only within one machine**.

**Data.** Demonstrations come from the scripted solver (`sim/solve_microwave.py`, `sim/generate.py`), rendered from the
camera above over the room splat (`render_va.py`) and encoded to VAE latents (`encode_va.py`). The final data set has 260
demonstrations (7 held out for validation):

| kind | demos |
|---|---|
| whole task | 59 |
| whole task with DART noise | 20 |
| starts in the middle of the task | 76 |
| recovery from a failed place | 46 |
| recovery from a missed door handle | 59 |

## 1. Zero-shot

- **Result:** no episode succeeded (0/21).
- **The LIBERO-long post-trained checkpoint** (`sim/EXPERIMENTS.md`, section 8b) never moved the pot and opened the door
  by at most 0.09°. Its actions were dominated by a constant forward-left drift.
- **Caveat:** this is not a test of whether the model can learn the task. The camera, table and action scales differ from
  LIBERO's.

## 2. First post-trained models (18-19 Oct)

The first runs differed in camera, attention window and data, and put the 7 joint targets and the gripper into the
action slots (the "joints layout"). Their protocols were not all the same, so only runs on one protocol are compared.

- **The only success:** the model with the camera pulled back, at step 1250: 1/8. Its door went past 90° in 0, 0, 1, 2,
  5, 0 of 8 episodes at steps 250..1500. Every other evaluation scored 0.
- **Same recipe plus 50 missed-handle retry demos:** never opened the door past 90° in 6 checkpoints × 8 episodes (at
  most 87.8°). Near-identical recipes gave 5/8 and 0/8 door openings at step 1250, so **retraining variance is large next
  to anything 8 episodes can show**.
- **Smoother solver demos:**
  - The smoother solver removed most large joint-space transfers: over the same 60 configurations, 59-60/60 solved vs
    53/60, and summed joint-space transfer 2967-3564° vs 5648°.
  - Trained on them, the model was 0/8 at step 250, as the earlier runs were.
- **Where the chain breaks:** over the 112 episodes on one protocol the door opened past 90° in 12, the pot was lifted
  ≥ 5 cm in 12, both happened in 2, the pot got inside in 1, and 1 succeeded. The chain mostly breaks at the pot grasp.

## 3. Action layout

- **Joints layout:** every early run put joint targets into slots that LingBot-VA's pre-training used for the end-effector
  pose.
- **Pretrain layout** (`--layout pretrain`): end-effector pose, joints and gripper in the pre-training's slots.
- **Before any training step,** the held-out action loss of the base model plus a fresh LoRA was 0.081 with the pretrain
  layout against 0.210-0.269 with the joints layout. The held-out video loss was similar (0.135-0.180).
- **Reading:** this is consistent with the pre-trained action pathway reading those slots as it was trained to. The
  joints layout threw that prior away.
- **Caveats:** different held-out sets and per-channel normalisations; step 0 only. Every later run uses the pretrain
  layout.

## 4. Executing the predicted pose by IK

- **Replay check on three solver demos:**
  - ManiSkill's `pd_ee_pose` fails all three (the hand is > 3 cm off from the first joint-space transfer on).
  - IK seeded near the joint target (`--exec ee_ik`) tracks as well as executing the joints (median 0.5 cm).
  - This shows the IK route executes without loss, not that it corrects a model's errors.
- **The combined run** (smoother demos + mid-task starts + failed-place retries, pretrain layout), six checkpoints × the
  same 8 starts:
  - 0/48 successes either way;
  - door ≥ 90° in 16/48 with IK vs 6/48 executing the predicted joints;
  - paired 15 vs 5 discordant, sign test p = 0.041.
  - Exploratory: door opening was not a stated endpoint. IK execution is used from here on.
- **At step 250** the predicted pose was slightly closer to the demonstration than forward kinematics of the predicted
  joints: median 2.7-4.6 cm vs 3.7-5.3 cm.

## 5. Under-training

The combined run was continued from 1500 to 15280 steps (stopped there).

- **Starts 0..7:** step 2500 0/8, 3500 0/8, **4500 2/8** (full task completions on video), 5500 0/8.
- **Fresh starts 8..31:** step 4500 solved 4/24 against 1/24 for the camera-pulled-back model at 1250 (one-sided Fisher
  p = 0.17). Its door opened in 22/24 vs 11/24 (p = 0.0007).
- **Step 14000** of the continuation: 4/8 on starts 0..7, and **8/24** on starts 8..31.
- **Reading:** under-training was a large part of the early zero rates.
- **Validation:** held-out robot video loss flattened at about 0.072 from step 4000, while the held-out human-video loss
  kept rising (section 18).

## 6. Camera

- **Camera pulled back** (0.3 m back, 0.1 m up): the first success (section 2). Every later run uses it.
- **Wrist camera added to the combined run:** 0 successes at every checkpoint. The door reached ≥ 90° in 12/48 episodes
  (16/48 without the wrist view). No gain shows at 8 episodes per checkpoint.

## 7. Starts in the middle of the task

- **Data:** the solver generated 76/80 mid-task demos, across four kinds of starting state.
- **First version of the "failed place" demo:** putting the pot down on the counter in front cannot work. Close to the
  robot's base the wrist touches the counter first, and the pot falls over.
- **Ablation:** mid-task starts were never removed from a recipe on their own.
- **Teacher-forced decision test** at the tuck after a place vs after a failed place:
  - With the true history, the model heads for the handle after a place and for the pot after a failed place: it follows
    the scene where the data contain the decision.
  - So the closed-loop failures to recover come with states or decision points the data lack, and with drift in the
    model's own history.

## 8. Retry demonstrations

- **Failed place** (`--place-short`, rewritten): 46/60 kept. The pot is let go half on the cavity's front edge; the arm
  tucks, returns, grasps it there and slides it in.
- **Missed door handle** (`--miss-back`): 59 demos. The gripper closes short of the handle, pulls on nothing, backs off and
  re-grasps.
- **Missed pot grasp** (`--pot-miss`): the gripper closes short and lifts on nothing, then retries. Only in the slow-demo
  data (section 9).
- **Ablation:** none of the three was removed from a recipe on its own. The handle-miss demos have a cost found later
  (section 17).

## 9. Slower demonstrations and a wrist camera

From base: demos at solver speed 2 instead of 3 (the same paths, 1.5× the steps), the wrist camera, and handle-miss and
pot-miss retries. The action length was drawn from 1..3 latent frames per chunk (`--chunk-max 3`). Trained 16000 steps,
then continued to 20000 with the lr falling linearly 1e-4 -> 1e-5 (111 h in total). Evaluated on the DGX Spark:

- **Starts 0..7,** queried every 8 steps, at 2000..16000: 0, 3, 1, 3, 2, 0, 4, 5 of 8.
  - Queried every 16 steps instead: 0, 1, 2, 2, 1, 1, 3, 3.
  - Same-rate test over 4000..12000: p = 0.30.
- **Continuation:** 18000 3/8; 20000 5/8.
- **20000 on starts 8..31:** **19/24** (95 % CI 58-93 %), against the combined run's 8/24 (one-sided Fisher p = 0.0016).
  - The two differ in many ways at once, so no single change is credited.
- **Weight averaging:** the exact average of five checkpoints (14000..18000) gave 4/8, the mean of its members. No gain.
- **No training collapse at 13000 steps.** Gradient norms fell, and validation video loss went flat while the training
  loss kept falling.
  - At constant lr the LoRA update random-walks: its distance from step 8000 grows like √n.
  - With the falling lr it moves less, in proportion to the lr.

## 10. Noise on the action history (the change that made it work)

**From base, 14000 steps, 40.7 h on the RTX 5090.** Data: the combined demos plus the handle-miss retries (the data
table above).
- **The noise:** in half of the samples, the past actions the model attends to are mixed as (1 − s)·a + s·ε, with s drawn
  per frame in [0.001, 0.5] (mean 0.25) (`--action-noisy-cond-p 0.5`).
- **Why this is new here:** LingBot-VA's recipe already noises the past video, more strongly; it leaves the past actions
  clean.

**Results.**

| | starts 0..7 at 2000..14000 | pick (rule) | starts 8..31 |
|---|---|---|---|
| with the noise | 1, 2, 0, 4, 3, 6, 5 of 8 | 12000 | 16/24 |
| without it (same data and settings) | 0, 0, 0, 4, 1, 1, 4 of 8 | 8000 | 5/24 |

- **Paired by start:** 13 starts were solved only with the noise and 2 only without (exact McNemar p = 0.007).
- **Without the noise,** 18 of the 19 failures are missed pot grasps, while the door, which never moves, opened in 21/24.
- **With the noise,** 7 of 8 failures are grasp misses (pot ≤ 5.4 cm), and one lifted the pot but never placed it.
- **Querying every 8 steps instead of 32 did not help:** 12/24 vs 16/24 (paired 2 vs 6, p = 0.29).

**Caveats.** One training seed per arm. Retraining variance measured in the first repository for one recipe (73 vs 49 %) is of a
similar size, so a second seed is needed to call this settled. The mechanism is in section 15.

## 11. Ego human video in training

The noise model's recipe plus 43 ego takes as video-only samples (30 % of samples). From base, 14000 steps:

- **Starts 0..7, 2000..14000:** 1, 1, 0, 4, 4, 3, 4 of 8.
- **The rule's pick (10000) on starts 8..31:** **11/24**. Paired against the noise model at 12000 (16/24): 4 vs 9, p = 0.27.
  **No gain is shown, and the direction is against it.**
- **The comparison is not matched.**
  - It was picked at 10000 steps rather than 12000, with no lr decay.
  - About 30 % of its updates were not robot updates.
  - "Co-training costs success" is therefore not established either.
- **On the human-video loss** it does what the others do not (section 18).

## 12. Continued training with lr decay

The noise model continued from 14000 to 20000 steps with the lr falling linearly 1e-4 -> 1e-5. Weights only: AdamW was
restarted, with a 50-step warm-up. Its last checkpoint was fixed before any result was seen. RTX 5090:

- **Starts 0..7:** 16000 7/8, 18000 6/8, **20000 8/8**.
- **20000 on starts 8..31:** **24/24**, both queried every 32 steps and every 8 (95 % CI 86-100 %).
  - Against the same model at 12000: 8 vs 0, p = 0.0078.
  - Against the slow-demo model at 20000: 10 vs 0, p = 0.002.
- **Validation losses** were again not a guide: robot video loss rose 0.0718 -> 0.0795 over the continuation, while the
  action loss fell 0.00106 -> 0.00090.
- **Confounds:** steps, lr decay and the optimizer restart changed together.
- **The slow-demo model's continuation** did not show a clear gain from the decay: 16000 vs 20000 was 12/24 vs 14/24
  (paired 3 vs 5, p = 0.73; RTX 5090).

## 13. The same weights on two machines

Identical code, splat, start list and LingBot-VA source on both machines. On the RTX 5090 the text encoder runs on the
CPU.

| model | DGX Spark | RTX 5090 | starts that flip |
|---|---|---|---|
| noise model, 12000, every 32 steps | 16/24 | 16/24 | 4 (2 each way) |
| slow-demo model, 20000, every 8 steps | 19/24 | 14/24 | 9 (7 Spark-only, 2 5090-only; p = 0.18) |

- **The order of the two best models reverses** between the machines, and neither difference is significant on either
  machine.
- **Bitwise determinism on one machine does not make a 24-start rate stable.**
- **The cause** (physics, rendering, kernels or the CPU text encoder) was not separated.
- **Consequence:** from here on, results are over 72 starts and from the RTX 5090 only.
- **At about equal steps:** the slow-demo model at 14000 (constant lr) scored 7/24 on the RTX 5090. That is behind the
  noise model's 16/24 (4 vs 13, p = 0.049, borderline with several tests run that day). It caught up only with the 6000
  further steps and the decay.

## 14. The 72-start confirmation (RTX 5090)

| model | 8..31 | 32..79 | 8..79 |
|---|---|---|---|
| noise model, continued to 20000 with lr decay | 24/24 | 47/48 | **71/72 (99 %, 93-100)** |
| noise model, 12000 | 16/24 | 39/48 | 55/72 (76 %, 65-85) |
| slow demos + wrist camera, 20000 (every 8 steps) | 14/24 | 29/48 | 43/72 (60 %, 48-70) |

- **Final model vs the same model at 12000:** 17 vs 1, p = 0.00014.
- **Final model vs the slow-demo model:** 28 vs 0, p = 7.5e-9.
- **The noise model at 12000 vs the slow-demo model:** 21 vs 9, p = 0.043. The noise model got there with about a third of
  the training time (40.7 h vs 111 h).
- **The final model's one failure** (start 50): it opened the door and lifted the pot to 19.6 cm, but never placed it.
- **Failures elsewhere:**
  - slow-demo model: 20 grasp misses, 9 lifted but not placed;
  - noise model at 12000: 12 grasp misses, 5 lifted but not placed.
  - The door reached ≥ 85° in all 144 episodes of the two references.

## 15. What the model looks at: a reliance diagnostic (`reliance_va.py`)

**Method.** Teacher-forced on three held-out demos. Replace the cached action history, or the cached video, with the
same demo's content from two chunks later, then measure how far the predicted tool path moves (median per chunk).
Re-seeding alone moves it 0.7-2.5 cm.

| model | action history swapped | video swapped | ratio |
|---|---|---|---|
| early combined run (no noise), 1500 | 36 / 24 / 36 cm | 11 / 8 / 12 cm | 2.0-2.8 |
| no-noise control, 14000 | 35.3 / 20.8 / 34.2 cm | 8.7 / 8.4 / 9.8 cm | 2.0-3.0 |
| noise model, 14000 (only the noise differs) | 25.4 / 14.8 / 25.6 cm | 21.3 / 16.8 / 20.9 cm | 0.94-1.10 |
| noise model, 20000 | 23.4 / 12.3 / 23.5 cm | 23.4 / 16.9 / 21.5 cm | 0.74-0.91 |

- **What changes:** without the noise, the model continues its own trajectory rather than looking at where the pot is.
  The noise roughly doubles the response to the video and cuts the response to the action history by about a quarter.
- **What does not change:** all three resolve the retry-vs-handle decision from the scene alike. So the difference is in
  where the arm goes, not in which phase comes next. That fits the no-noise failures being grasp misses.
- **Reading:** this is the copycat failure of imitation from histories, with the same remedy as ChauffeurNet's past-motion
  dropout. It does not exclude exposure bias. Literature: `literature/action_history_noise.md`.
- **One episode (start 25), qualitative:**
  - The no-noise control never grasps the pot. At 22 s its imagined frames show the pot inside the cavity while the
    camera shows it still on the counter, and it goes on to close the door.
  - The noise models place the pot, with imagined and seen frames agreeing.

## 16. Hand-over rescue (`handover_va.py`)

**Question:** can the final model rescue an episode in which the no-noise model's grasp failed?

**Protocol.**
- The no-noise control drives starts 8..31 until its first pot grasp fails: the pot has risen less than 5 cm 32 steps
  after the gripper closed. That happened on 17 of 24 starts.
- At the end of that chunk another model takes over, with 1000 steps left. Either it inherits the same history (replayed
  into its cache, bit-exact on all 68 runs), or it starts with an empty cache (`--fresh`).

| who continues | success | pot re-lifted > 5 cm |
|---|---|---|
| the no-noise model itself | 0/17 | 0/17 |
| final model, inheriting the history | 0/17 | 0/17 |
| final model, empty cache | 3/17 | 10/17 |
| noise model at 12000, inheriting the history | 0/17 | 1/17 |

- **Empty cache vs inherited history (final model):** re-grasp 10 vs 0 (p = 0.002); success 3 vs 0 (p = 0.25). The
  successes were starts 19, 22 and 25.
- **The inherited history stops even a model that can re-grasp from the same state.** With the history of a
  completed-looking grasp, it goes on to the next phase (closing the door).
- **Among the re-grasps that did not succeed,** one (start 27) put the pot inside but tilted (it fails the flat-placement
  check), and the door was not shut in time.
- **Caveat:** neither model was trained on recovering from a missed pot grasp.

## 17. Planned door misses

**Census.** 59 of the 253 training demonstrations miss the door handle on purpose (23 %). In the evaluation videos, at
6.2 s the arm is still at the handle with the door shut in:

| model | episodes with a first pull that failed |
|---|---|
| no-noise control, 8000 (8..31) | 9/24 (3 more never opened the door) |
| noise model, 12000 (8..79) | 2/72 |
| noise model, continued to 20000 (8..79) | 12/72 |

Doors that open at the first pull are open by about 5 s.

- **Longer training brought the scripted miss back:** 20000 vs 12000 on the same starts, 10 vs 0, p = 0.002, i.e. toward
  the data's 23 %.
- **The miss is planned, not a surprise.**
  - The imagined frames show the door shut at the same moments. Dense frames of three episodes show the imagined door
    staying shut through the pull.
  - The pot-grasp misses are different: there the imagination shows success (section 15).
- **Why success rates do not show it:** the final model still completes these episodes; the retry costs about 3 s.
- **Proxy and limits:** "door shut at 6.2 s" was checked frame by frame for three starts only; one machine.
- **Literature:** `literature/recovery_without_imitating_failures.md` covers how other work teaches recovery without
  supervising the failure.

## 18. Forgetting: the held-out human-video loss

**The first runs.** Robot-only training raised the validation loss on held-out ego takes, each run on its own held-out
set:
- the camera-pulled-back model 0.240 -> 0.284;
- the same with handle-miss demos 0.240 -> 0.279;
- the combined run 0.279 -> 0.323 over its continuation.

With 30 % ego video in an early run, it fell 0.198 -> 0.131.

**The later runs** (same held-out takes, base 0.240):

| run, human-video loss at step | 0 | 2000 | 6000 | 10000 | 14000 | 20000 |
|---|---|---|---|---|---|---|
| noise model (continued to 20000 with lr decay) | 0.240 | 0.283 | 0.314 | 0.342 | 0.362 | 0.410 |
| same model, robot-video loss | 0.171 | 0.077 | 0.067 | 0.068 | 0.072 | 0.080 |
| no-noise control | 0.240 | 0.283 | 0.316 | 0.345 | 0.364 | -- |
| noise model + 30 % ego video | 0.240 | 0.152 | 0.184 | 0.217 | 0.250 | -- |
| slow demos + wrist camera (continued to 20000) | 0.240 | 0.270 | 0.294 | 0.317 | 0.327 | 0.358 |

- **Steady drift.** From about 6000 steps on, the human-video loss rises roughly linearly, by about +0.007 per 1000 steps.
  The lr decay did not slow it.
- **The action-history noise does not change it:** the two runs differ by at most 0.005.
- **The two losses come apart.**
  - Held-out robot-video loss bottoms out at 6000-8000 steps (0.067) and then rises.
  - Over the same steps, closed-loop success rose from 55/72 (12000) to 71/72 (20000).
- **With 30 % ego video, the level stays at the base over the trained range.**
  - It stays at or below the base for the 14000 trained steps (at most 0.250 vs 0.240), against 0.362 for the robot-only
    run at the same step.
  - But it rises at every checkpoint after step 2000, with no plateau: from 6000 to 14000 it gains +0.0083 per 1000 steps
    vs +0.006 for the robot-only run.
  - It crosses the base between 13000 (0.243) and 14000 (0.250).
  - This held-out set comes from the same ego data the run trains on, so it mixes forgetting with learning and then
    overfitting that data. It is not a clean forgetting measure for this run.
- **What is not known:** whether the drift costs anything in the twin. No out-of-distribution probe exists yet.
  Literature: `literature/forgetting.md` and `literature/kl_to_base.md`.

## What did not work

- **fp8 training** (torchao `Float8Linear` on the frozen base, without `torch.compile`) was slower: 26 s vs 17.6 s per step,
  peak memory 23.5 vs 22.2 GB.
- **ManiSkill's `pd_ee_pose`** as the executor for predicted poses (section 4).
- **The first "failed place" demo:** putting the pot down in front of the robot (section 7).
- **Wrist camera added to the combined run** (section 6), and **ego human video** in training (section 11): no gain shown.
- **Averaging checkpoint weights** (section 9).
- **Querying more often:** every 8 steps instead of 32 did not help the noise model (section 10). For the combined run at
  14000, every 8 steps scored 1/8 against 4/8 at both 16 and 32.
- **Moving the robot's base:** the task solves only in a narrow band of base positions. Elsewhere the 90° door-pulling arc
  ends at a joint limit, with no IK path to release.

## Limitations

- **One training seed per recipe.** Retraining variance was not measured for any LingBot-VA recipe.
- **One machine for the final numbers** (RTX 5090). Counts moved by up to 5/24 between machines (section 13).
- **Steps and lr decay changed together** in both continuations, with the optimizer restart as a third change.
- **Several paired tests on the same day:** p values near 0.05 are borderline.
- **Simulation only;** one task, one scene, one camera. The pot's starts lie within the range of the demonstrations.
- **No recipe component was removed on its own:** mid-task starts, retry demos, pretrain layout, full-window attention.
- **Proxies:** the planned-miss census is a timing proxy. The human-video loss is a proxy for the video prior; its cost
  in the twin is unmeasured.
