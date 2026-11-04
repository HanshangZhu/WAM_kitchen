# Human ego data -> robot actions: Dyna, HumanEgo, SuperMimic (4 Nov 2026)

Written to check three claims that are easy to make together: "human-demo to robot-action commutativity needs scaled
data; HumanEgo gives direct supervision when data is scarce; SuperMimic-style strong priors warm up early exploration
and value shaping". Section numbers and verdicts below refer to them.

All primary pages were fetched and read in full: the dyna.co research index and posts, arXiv HTML for HumanEgo v3
and the PI paper, the arXiv PDF for SuperMimic v1, and the PI blog. Search summaries were used only to find the pages.
Tags on numbers:
- [text]: stated in the prose.
- [svg]: a value label printed in the page's chart (exact, but shown only in the figure).
- [svg~]: rebuilt from chart point coordinates, about +-0.005.
- [derived]: arithmetic done for these notes.
- fig-only: shown in a figure and not readable from the source.

None of the sources uses the word "commutativity" or claims that human and robot data become interchangeable.

## 1. Dyna Robotics posts

Research index (dyna.co/research), with the dates the index gives:
- 09.28.26: *Dyna-2.1: A Physical Agent for End-to-End Workflows* (dyna.co/dyna-2.1). The post's byline says 29 Sep 2026.
- 08.27.26: *Not Just a Model, But a Product* (deployment ops; no human-data science).
- 08.17.26: *Training Dyna-2 at million-hour scale, repeatably* (data and training infrastructure only).
- 08.15.26: *Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models* (dyna.co/dyna-2). The byline says "August 2026".
- 11.20.25: *Dyna-1 Pre-Training: Zero-Shot Dexterity Is Here* (a short update with no human data).
- 10.15.25: *Open-World Dexterity and Live Demos around the World*.
- 06.24.25: *Dynamism v1 (Dyna-1)* (reward model, napkin folding).

The separate "Blog" section (/news) has only business posts.

**Which two posts matter here.** The most likely pair is Dyna-2 (human-video scaling) and Dyna-2.1 (human data made usable for
robots, and RL seeded with human motion). The second is ambiguous: the 08.17 infrastructure post is also "about scaling",
but only about I/O and optimizer engineering. Both posts are company reports, not peer-reviewed, with "Dyna Robotics" as
the only author.

### 1a. Dyna-2 (dyna.co/dyna-2)

**Method.**
- A world-action model on a video-diffusion backbone: a mixture-of-transformers with separate video and action DiT stacks,
  trained with flow matching.
- In the scaling-law variant, video and action are co-trained as two marginal velocity fields. The action head never takes
  noisy video, so at inference the policy "neither generates nor attends to predicted future video" (§2).
- Human actions are pseudo-labels from 3D hand pose (§3, "Pre-training data").
- There is no visual or kinematic gap reduction, and no robot data in pre-training.

**Data.**
- Nested human-video subsets of exactly 1k / 10k / 100k / 1M hours, with the same source proportions.
- Training and eval configs are identical across rungs. Compute and model size are not scaled: "We leave compute and
  model-size scaling experiments for future work."
- Zero-shot robot eval: 39 tasks on two YAM bimanual platforms (12 internal, 27 from xdof ABC).
- On-robot post-training: 14 tasks with at most 10 h of robot data each (Bottle Cap about 10 min), 10 trials per task
  (12 for language), blind evaluators.

**Results.**

| Measure | 1k | 10k | 100k | 1M | Fit / note |
|---|---|---|---|---|---|
| Held-out human MSE (Fig 5) [svg] | 0.062 | 0.057 | 0.056 | 0.054 | 0.0691*D^-0.0184, R^2 0.919 |
| Held-out human acc@0.5 [svg] | 0.40 | 0.44 | 0.45 | 0.47 | 0.357*D^0.0203, R^2 0.865 |
| Held-out human acc@0.1 [svg] | 0.017 | 0.021 | 0.024 | 0.026 | 0.0116*D^0.0606, R^2 0.926 |
| **Zero-shot robot MSE, all 39 (Fig 7)** [svg] | 0.180 | 0.174 | 0.124 | 0.117 | 0.306*D^-0.0713, R^2 0.884 |
| Zero-shot robot acc@0.5, all 39 [svg] | 0.067 | 0.074 | 0.136 | 0.159 | 0.0241*D^0.139, R^2 0.918 |
| Zero-shot MSE, dyna 12 / xdof 27 [svg~] | .186/.178 | .182/.172 | .148/.112 | .126/.108 | both monotone |
| On-robot mean normalized score, 14 tasks (Fig 8) [text] | 20 % | 28 % | 45 % | 53 % | best on 9/14 at 1M |

- The held-out human MSE improves only 12 % across a 1000x data ladder (the text says "rises 51 %" for acc@0.1 vs 12 % for
  MSE).
- The robot curve is nearly flat from 1k to 10k, drops at 10k -> 100k, then flattens. The authors call this an "inflection".
- Per-task on-robot results (Fig 8 [svg], 10 trials each). Only 6 of the 14 tasks are non-decreasing [derived]. 1M is
  below 100k on four tasks: Rope Tie 90 -> 40, Food Scooping 80 -> 50, Tote 40 -> 30, Pick & Place 4.2 -> 3.9.
  - Lockbox 0 / 0 / 0 / 90 %.
  - Bottle Cap 10 / 10 / 40 / 50 %.
  - Drink retrieval 58 / 75 / 83 / 83 %.
- Objective ablation (Fig 10 [svg~]). Zero-shot robot MSE at 5k / 50k / 100k action-labelled hours:
  - action-only: 0.27 / 0.35 / 0.25;
  - joint (video + action): 0.16 / 0.17 / 0.17;
  - +video co-training: 0.18 / 0.13 / 0.12.
  - Text: joint beats action-only "on 39 of 39 tasks at every action scale". Video co-training "is not advantageous at
    small scale (i.e., 5k hours), but the gap grows as data scale increases."
- "Video is the new scaling axis" (Fig 11).
  - With action-labelled data fixed at 50k h, video-only hours 0 / 1k / 10k / 50k give robot MSE 0.34 [svg] / 0.18 [svg~]
    / 0.15 [svg~] / 0.12 [svg].
  - With action data fixed at 250k h, video-only hours 0 -> 750k give 0.10 -> 0.084 [svg].
  - Fig 12: with more video, held-out human error is 104 % of the no-video arm, while robot error falls to 34 %.
- Other claims, outside the scaling study:
  - WAM vs Dyna-1 VLA: 1.55x success rate, 1.12x grade (7 tasks x 3 checkpoints).
  - Customer-site pass rate 46 % -> 87 %.
  - Language following 0.35 -> 0.67 -> 0.96.

**Quotes.**
- §3.2: "We empirically observe an inflection point from 10k to 100k hours of pre-training, suggesting with sufficient
  coverage, cross-embodiment knowledge transfer may emerge from just scale alone."
- §3.3: "Several tasks appear to require a threshold amount of pre-training before they become solvable at all."
- §3.3: "we deliberately do not perform human-robot alignment or co-training"
- §3, data: "...from which we derive the pseudo-action supervision the action stream trains on: wrist poses for end-effector
  trajectories, and a continuous grasp signal derived from the thumb–index aperture."
- §3, metrics: "Scaling claims can be artifacts of the metric used to make them". This is why they report two continuous
  and two thresholded metrics.

**Complications.**
- One training run per rung, with no seeds. The 10-checkpoint averaging is stated for the human metrics only.
- Compute is fixed, so the smaller rungs see their data many more times. The curve mixes "more data" with "less
  overfitting", and the post's own Fig 10 shows severe overfitting for action-only.
- A power law and an "inflection" are both claimed from four points; four points cannot tell them apart.
- Absolute zero-shot transfer stays weak. Robot acc@0.5 is 0.159 at 1M, against 0.47 on held-out human data. This is a
  trend across the embodiment gap, not human data becoming equivalent to robot data.
- The Fig 11 "0-video" arm is action-only. Most of the 0.34 -> 0.12 drop is switching on the video objective (joint alone
  is about 0.17 at 50k, from Fig 10). The gain from video-only hours themselves is about 0.17 -> 0.12 [svg~].

### 1b. Dyna-2.1 (dyna.co/dyna-2.1)

**Method.**
- Taku semi-humanoid, with a hierarchy:
  - an RL whole-body controller at 100 Hz;
  - an improved Dyna-2 policy that outputs task-space targets;
  - a VLM orchestrator.
- Human, teleoperation and UMI data share one interface, the "Unified Robot Representation": wrist, elbow, chest and
  footprint poses. Converting human motion into it is "a retargeting step that stays in task space" (§3.1.1).
- The policy is pre-trained on 1M h of human video "mixed with robot data from our fleet" (§3.1.1). This differs from
  Dyna-2's human-only scaling study.
- §3.1.2 bears directly on (c): "Seed reinforcement learning with human motion."
  - "Rather than overly relying on exploration and manual reward tuning, we put our massive reservoir of URR data to use,
    specifically as pose targets alongside synthetic targets to inject natural human pose priors into control learning."
  - "Empirically, we observed that the growing human prior not only induces faster convergence, but also increases the
    quality of our controls in offline metrics and teleoperation tests."
- **No numbers, curves or ablation are given for this claim.** It concerns the tracking controller, not the manipulation
  policy, and there is no value-function measurement.

## 2. HumanEgo

**Citation.** Zhi Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, Yiannis Aloimonos (Univ. of Maryland).
*HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos*. arXiv 2605.24934: v1 24 May 2026, v3
14 Sep 2026. No venue is listed (UNVERIFIED). Code: github.com/TX-Leo/HumanEgo. This is the only paper that matches the
name; no competing candidate was found.

**Method.**
- Aria Gen1 glasses record the human demonstrations.
- The human arm is inpainted out (SAM2 + LaMa), and a virtual gripper and object keypoints are rendered in its place.
- Hands and objects become "Interaction-Centric Tokens": each entity's pose plus both hands' poses in that entity's frame.
- A flow-matching policy is trained with three dense auxiliary losses: future 6-DoF object motion, 2D traces, and
  latent ICT state K steps ahead.
- **Zero robot data.** At deployment only the input front-end is swapped: depth back-projection and forward kinematics.

**Action labels (b).**
- These come from retargeting the hand pose. Aria MPS stereo 3D hand keypoints are lifted to the world frame.
- The thumb–index pair is treated as a virtual parallel-jaw gripper:
  - position = the fingertip midpoint;
  - orientation = a Gram–Schmidt frame built on the MCP joints;
  - grasp = the normalized thumb–index distance, binarized at deployment (§3.3, App E.3).
- There is no pixel-space video prediction.

**Data.**
- About 30 min of human video per task, about 45 demos at about 40 s each.
- Four tasks, 40 trials each, on Trossen WidowX arms.
- Baseline: ACT on 30 min of teleoperation.

**Results [text].**
- Mean over 4 tasks: 92.5 % at 30 min and 75.0 % at 15 min, vs ACT at 30 min 51.2 %. The headline "+41 pp" is
  92.5 - 51.2. The arXiv abstract page says "by 41%"; the v3 text says "41 percentage points".
- Per task: Downstack Cups 87.5 %, Water Flowers 95 %, Serve Bread 95 %. Adjust Table is 92.5 % [derived].
- The five human-video baselines (EgoZero, Point Policy, ZeroMimic, Track2Act, SPOT) score 1.9-45 %.
- Scaling on Serve Bread (Fig 5):
  - about 7 min -> 50 %, 30 min -> 95 %;
  - at 8 min, with vs without the auxiliary losses: 57.5 vs 37.5 %;
  - "At 18 minutes and beyond the two variants coincide";
  - the same policy class trained on 30 min of teleoperation gets 65 % vs 95 % on human data.
- Representation ablation (Water Flowers, 30 min):
  - raw human RGB 7.5 %, inpainting + keypoints 20 %, robot RGB 32.5 %;
  - raw RGB + ICT 85 %, ICT without the image 75 %, full system 95 %.
- Auxiliary losses at 15 min: +17.5 / +5 / +12.5 pp each, +25 pp together. The baseline is fig-only.
- Hand tracker swap: 95 -> 45 -> 32.5 -> 0 % (Aria-MPS -> WiLoR -> HaMeR -> MediaPipe).
- Nine OOD conditions (viewpoint, lighting, Franka, UR10): 85-91.25 %.
- Co-training, App H.2 (Serve Bread, 30 min total): 0 / 25 / 50 / 75 / 100 % human gives 65 / 72.5 / 77.5 / 90 / 95 %.

**Quotes.**
- §1: "robots should not imitate the human body, but recover the task-relevant interaction geometry that transfers across
  bodies."
- §3.4: the shared encoder is "a multi-task regularizer that curbs overfitting, with the largest gains in the low-data
  regime"
- App H.2: "We find no “sweet spot” where mixing in robot data outperforms the human-only condition".
- §5: "few-shot learning plateaus at ∼1 cm precision"

**Complications.**
- HumanEgo is not "human data supervising a robot policy when robot data is scarce". Robot data is zero, and what is scarce
  is human minutes.
- Transfer works at minutes scale through hand-engineered alignment, which runs against "transfer needs scale".
- Scope is limited: parallel-jaw arms only, a quasi-static scene, and dependence on Aria stereo tracking.
- Each comparison rests on 40 trials and one policy. The H.2 claim that 7.5 min of teleop "actively erodes" the human data
  rests on 90 vs 95 %, which is 2 trials.
- Internal inconsistency:
  - H.2 says about 45 teleop episodes in 30 min;
  - §4.1 and App D give 60-70 s per teleop demo, which means about 26-30 [derived].

## 3. SuperMimic

**Citation.** Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath (UC
Berkeley). *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. arXiv 2610.28480,
20 Oct 2026. CoRL 2026 (Oral), per the arXiv comment and the PDF footer. This is the only paper found under the name.

**Method.**
- The policy tracks G1-retargeted human mocap clips.
- One deterministic world model of the shared dynamics is fit *online*, from the policy's own simulator transitions
  (MuJoCo Warp, k = 8 windows).
- The policy is trained by BPTT through H = 32 imagined steps, which gives analytic policy gradients.
- A critic, fit to GAE returns on the imagined rollouts, bootstraps the return beyond H.

**Data.**
- 264.56 h of BONES-SEED mocap (129,514 clips, the public subset of SONIC's 611 h).
- LAFAN1 (40 clips, 2.45 h) for the controlled tests.
- Sim only. The real G1 demos (4 motions) are qualitative.

**Results [text].**
- vs SONIC: within 3 h on 1 H100, SuperMimic beats the return of SONIC's released checkpoint (128 L40 x 168 h, about
  21,000 GPU-h). That is more than 1000x fewer GPU-hours.
- Failed motions: 25 for SuperMimic after 3 h on 1 GPU, 0 after 9.8 h on 2 GPUs, against 179 for SONIC.
- vs PPO in the simulator on LAFAN1: SuperMimic reaches return 0.82 in 1.2 h. PPO-Sim never reaches it in 18 h, ending at
  0.60 / 0.70 / 0.72 on 1 / 4 / 8 GPUs.
- vs PPO on the same world model: SuperMimic needs about 64x fewer imagined steps to reach PPO-WM's best return.
- Rollout speed: the world model is more than 10x faster than the simulator, up to 62x at 64k environments.
- 8-step world-model error stays at 0.007-0.011 from 1 to 264 h of motion.
- Critic ablation:
  - without a critic, training fails at H <= 16;
  - with a critic, H = 16 matches the critic-free H = 32;
  - under 0.5-4 m/s pushes, only the critic policy settles (in 0.5-2.4 s).

**On claim (c).**
- **SuperMimic does not use a strong prior to warm-start exploration or the value function.**
  - The world model starts untrained and is fit in the loop. §2: "SuperMimic fits its world model during training on data
    its own policy collects", in contrast to RWM, which pretrains it.
  - The policy is deterministic. The action noise "serves only the model fit and is optional" (§3.1).
  - The critic is learned from scratch on imagined rollouts.
- The speed-up is attributed to three things (§4.2): model accuracy that holds as data grows, fast rollouts, and
  first-order gradients ("the gain is not just rollout speed").
- The only "human prior" is the mocap reference used as a dense tracking reward. The authors credit that dense reward for
  stability: "We ran several random seeds and found training very stable, which we attribute to the dense reward that
  motion tracking gives at every step. The figures therefore do not report seed variation." (§4)
- Caveats from the paper itself: the comparison with SONIC is not like-for-like (2.3x less data, different GPUs), and "The
  benchmark motions are training motions for both methods" (App F.2).

## 4. Where "transfer emerges with scale" actually comes from

**Kareer, Pertsch, Darpinian, Hoffman, Xu, Levine, Finn, Nair.** *Emergence of Human to Robot Transfer in
Vision-Language-Action Models*. arXiv 2512.22414 (27 Dec 2025). PI blog, 16 Dec 2025: pi.website/research/human_to_robot.

**Method.**
- π0.5 + ego: human data is treated as one more embodiment.
- Human actions are relative end-effector poses, taken from 3D hand keypoints (palm, middle and ring fingers). There is no
  gripper action for humans.
- Subtask language labels are also used.
- Human data is co-fine-tuned 50/50 with the nearest-neighbour robot task. There is no explicit alignment.

**Data.** Only about 14 h of human data (3 + 3 + 3 + 5 h, §IV-A). The scale axis is the *diversity of robot
pre-training*, not the amount of human data:
- 0 %, 25 %, 50 %, 75 % and 100 % of the scene-task combinations;
- plus the full cross-embodiment mix ("100 % + X-emb").

**Measured as.** Task score with human data minus without (Fig 2, fig-only per task), plus a t-SNE overlap of human and
robot embeddings (Fig 5, qualitative).

**Results with human data [text].**
- Spice 32 -> 71 %, Dresser 25 -> 50 %, Bussing 53 -> 63 %, egg-sorting accuracy 57 -> 78 %.
- No gain at 0 / 25 % pretraining. Gains appear at 75 / 100 %.
- Sec VI: "as pretraining diversity grows past a critical threshold, transfer emerges."

**Nearest thing to "interchangeable" (§V-D).**
- At full pretraining, human data was "nearly as effective as" target-robot data for Sort Eggs and Dresser.
- It was not for Bussing: "target robot data was more effective than human data alone (25% vs 65%)". Which arm each
  number belongs to is fig-only.
- Human -> ARX transfer behaves like UR5 -> ARX robot transfer (400 demos, 7.45 h): both beat the baseline, and neither
  matches target-robot data.
- §II: "such methods are generally brittle at a small scale. As such, they often rely on some form of alignment to work
  well". This sentence reconciles PI with HumanEgo.

## Reading: the three claims

| Claim | Verdict |
|---|---|
| Human -> robot "commutativity" needs scaled data | **Partly.** Transfer without alignment grows with scale in PI (robot-pretraining diversity, threshold-like, one fine-tune per level) and in Dyna-2 (human hours, four single-run rungs, an "inflection" at 10k -> 100k). Neither shows interchangeability; Dyna-2's zero-shot robot acc@0.5 is 0.159 vs 0.47 on human data. HumanEgo is a counterexample on the "needs scale" half: minutes suffice with engineered alignment. |
| HumanEgo gives direct supervision when data is scarce | **Wrong framing; the mechanism is right.** It is robot-data-free, with action labels retargeted from stereo 3D hand pose. Its low-data gains come mostly from the representation (ICT: 7.5 -> 85 %) and from auxiliary losses that matter only below about 18 min. |
| SuperMimic-style strong priors warm up exploration and value shaping | **Not supported by SuperMimic.** Its gains come from a learned differentiable model, analytic gradients and critic bootstrapping, with no warm-start and no exploration story. The nearest source is Dyna-2.1 §3.1.2 (human-pose targets give "faster convergence" for the RL controller), which is unquantified. |

## What to correct before citing

1. Replace "commutativity" with what was measured. For example: "human-to-robot transfer without explicit alignment
   appears only once pre-training is large (PI: robot-data diversity; Dyna-2: human-video hours, offline zero-shot action
   error and post-trained success)". Cite the PI paper as the origin of the emergence claim; it is reference [36] in
   Dyna-2.
2. Describe HumanEgo as "robot-data-free transfer from about 30 min of human video per task, via retargeted hand-pose
   actions plus an interaction-centric representation". Do not describe it as supervision for a data-scarce robot policy.
   This also places it as the small-scale, explicit-alignment counterpart to the emergence results.
3. Drop "warm up exploration and value shaping" for SuperMimic, or re-attribute it. Accurate options:
   - "a learned differentiable world model trained in the loop, with analytic gradients, cuts GPU-hours by more than 1000x";
   - "human motion priors speed RL controller convergence (Dyna-2.1, qualitative)".
4. When quoting Dyna-2 numbers, add three qualifiers: company blog, single run per rung, compute held fixed.
