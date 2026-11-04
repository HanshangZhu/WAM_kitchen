# Fine-tuning without forgetting -- literature, 2025-26 (4 Nov 2026)

Question: can the model gain the task without forgetting its video prior? Our measurement: the held-out human-video loss
rises from 0.240 (base) to 0.410 (the final model, 20000 steps; `../EXPERIMENTS.md`, section 18).

36 papers and blog posts were checked on arXiv, venue pages or the blogs. Numbers were read in the text, except:
- [ext]: pulled by a summarising fetch; spot-check before quoting.
- fig-only: shown only in a figure, not read.
- 2026 arXiv items without a venue are marked "arXiv".

## Consensus (several independent groups)

1. **Forgetting tracks how far the model's outputs move from the base, not how many parameters change.**
   - Kalajdzievski (arXiv 2401.05605): forgetting is about linear in the fine-tuning loss.
   - RL's Razor (Shenfeld, Pari, Agrawal, ICLR 2026, 2509.04259): forward KL to the base, on the new task, predicts
     forgetting. R^2 0.96 on a toy task, 0.71 on LLMs.
   - Huan et al. (ICML 2026, 2507.00432), same math data: SFT has KL 0.283 and IFEval falls 69.2 -> 42.3; RL has KL
     0.019 and IFEval is 70.0.
   - Weight change, rank and sparsity were tested as predictors and lost.
2. **Replaying old data is the most reliable fix; a small fraction does most of the work.**
   - Ibrahim et al. (TMLR 2024, 2403.08763): about 5 % replay for a weak shift, about 25 % for a strong shift.
   - Also: RT-2 co-fine-tuning; UWM (RSS 2025); VPP (ICML 2025); Zhu et al. (arXiv 2605.26820), where pi0.5 learning 10
     real tasks in sequence scores 97.2 with replay vs 31.4 without.
   - Thinking Machines blog (Oct 2025): IF-eval 85 -> 45 with no mix, 79 with a 70/30 mix.
   - Replay alone rarely closes the gap fully.
3. **Interpolating base and fine-tuned weights is a cheap trade-off, and sometimes beats both ends out of
   distribution.**
   - WiSE-FT (CVPR 2022).
   - Lin et al. (EMNLP 2024): beat L2, distillation, LoRA and replay on the RLHF alignment tax.
   - RETAIN (Yadav, Zhou, Wagenmaker, Pertsch, Levine; ICLR 2026, 2512.08333): alpha-interpolation on pi0 / pi0-FAST,
     alpha in {0.25, 0.5, 0.75}, beats both ends OOD and keeps generalist skills [ext]. The closest analogue to our
     setup.
   - ReVLA (ICRA 2025).
4. **Training on the model's own outputs (on-policy) forgets less than SFT on outside targets.** RL's Razor; Retaining
   by Doing (Chen et al., ICML 2026, 2510.18874); Yang et al. (ACL 2024); Shenfeld et al. (ICML 2026, 2601.19897).
5. **LoRA reduces forgetting but does not prevent it, and high rank weakens the protection.**
   - Biderman et al. (TMLR 2024): on math, LoRA forgets about as much as full fine-tuning.
   - Shuttleworth et al. (NeurIPS 2025): "intruder" singular directions cause it. Scaling the top direction to 0.3 cut
     forgetting 33 % for a 0.1 % task loss.
   - Our rank 128 is not a safeguard.
6. **Fine-tuning a VLA on robot data alone wipes out its vision-language ability.**
   - ChatVLA (arXiv 2502.14420): 0 on every VQA benchmark.
   - OpenVLA and pi0.5 lose most of it (VLM2VLA, ICLR 2026, Table 1).
   - Knowledge Insulation (Driess et al., NeurIPS 2025) was built to stop this, and calls freezing the backbone "not
     viable".

## Contested or single-paper

- **"RL forgets less" as a general law.** Contradicted for tasks learned in sequence ("RL Forgets!", arXiv 2607.04364,
  abstract only). In Retaining by Doing, the off-policy targets come from a 70B model.
- **Spurious forgetting** (Zheng et al., ICLR 2025): the drop is lost alignment, not lost knowledge, and recovers in a
  few hundred steps. One paper; freezing more layers does not help monotonically.
- **KL to the base as a regulariser** (Riemer et al., arXiv 2512.22337, [ext]; the numbers were later read in the text, see kl_to_base.md): KL alone learns less or still forgets;
  KL plus replay is best.
- **Replay also improves the new task itself.** Kotha & Liang (arXiv 2603.04964), LLMs only.
- **Video world models.** No controlled study was found that measures how robot fine-tuning degrades a video-diffusion
  prior.
  - DreamGen (CoRL 2025: LoRA r4 "to mitigate forgetting") asserts it without measuring it.
  - So does LingBot-VA 2.0 (arXiv 2607.08639: trains from scratch "to avoid forgetting").
  - Our 0.240 -> 0.410 is itself uncommon evidence. "Not found" is a search result, not proof of absence.

## What applies to our setup (our reading)

- **The LoRA adapter is saved on its own** (the final model's adapter checkpoint), so the interpolation sweep needs no
  training. Use W_base + lambda (alpha/r) B A for lambda in {0, .25, .5, .75, 1}. Keep the fully trained
  action_embedder and action_proj_out as they are: lambda = 0 is then "base + our action interface", not the raw base.
- **Knowledge-Insulation-style stop-gradient does not map onto LingBot-VA.**
  - The open-source model is a joint DiT: video and action tokens share every attn / ffn weight, so blocking the action
    loss from "the video stream" would also block it from almost all the weights that learn actions.
  - It would need token-routed LoRA (separate adapters for video and action tokens).
  - It does apply directly to mixture-of-transformers WAMs such as Fast-WAM.
- **Replay fraction.** Sim kitchen -> ego human video is a strong shift, so Ibrahim's ~25 % is the reference, not 10 %.
- **"Human co-training costs task success" is not established.**
  - The rule's pick from the 30 %-human run was at 10000 steps, with no lr decay and about 30 % fewer robot updates.
  - Paired against the noise run's 12000: 4 vs 9, exact McNemar p = 0.27.
  - A run matched on the number of robot updates is needed before calling it a trade-off.
- **Retention has no shown value yet.** The task is already at 99 %. A lower human-video loss only matters if keeping
  the prior buys generalisation, so any of these tests needs an OOD probe (new layout, lighting, object pose); that is
  what RETAIN measured.

## Our own curves (validation losses logged during training)

| run | 0 | 2000 | 6000 | 10000 | 14000 | 20000 |
|---|---|---|---|---|---|---|
| action-history noise (continued to 20000 with lr decay), human video | 0.240 | 0.283 | 0.314 | 0.342 | 0.362 | 0.410 |
| same model, robot video | 0.171 | 0.077 | 0.067 | 0.068 | 0.072 | 0.080 |
| no-noise control, human video | 0.240 | 0.283 | 0.316 | 0.345 | 0.364 | -- |
| noise + 30 % ego video, human video | 0.240 | 0.152 | 0.184 | 0.217 | 0.250 | -- |
| slow demos + wrist camera (continued to 20000 with lr decay), human video | 0.240 | 0.270 | 0.294 | 0.317 | 0.327 | 0.358 |

Readings (descriptive; one seed each):
- **Steady drift.** From ~6000 steps on, the human loss rises about linearly, roughly +0.007 per 1000 steps. The lr
  decay over the continuation (14000 -> 20000) did not slow it.
- **The action-history noise does not change forgetting.** The noise and no-noise runs differ by at most 0.005 at every
  checkpoint (12000: 0.361 vs 0.356).
- **Held-out robot video bottoms out at 6000-8000 (0.0665) and then rises**, while closed-loop success keeps rising
  (12000: 55/72, 20000: 71/72). Both validation losses drift away together as success improves.
- **30 % human video gives an offset, not a slope change.** Full series (every 1000 steps): 0.240, 0.157, 0.152 (minimum,
  step 2000), 0.157, 0.167, 0.177, 0.184, 0.196, 0.204, 0.209, 0.217, 0.220, 0.233, 0.243, 0.250 (14000). Rises at every
  checkpoint after step 2000, with no plateau.
  - Level: over the 14000 trained steps it stays at or below the base (at most 0.250 vs 0.240),
    while the robot-only run reaches 0.362 at the same step. By level, the co-trained model has not forgotten over that
    range.
  - Slope: from 6000 to 14000, +0.0083 per 1000 steps vs +0.006 for the robot-only run. It crosses the base between
    13000 (0.243) and 14000 (0.250); at 20000 a straight-line extrapolation gives ~0.30 (not measured).
  - The loss first falls below the base (0.152 at 2000), then rises from 6000 to 14000 at about the same rate as without
    human video.
  - In this run the human validation loss mixes forgetting with overfitting to the human training takes, so it is not a
    clean forgetting measure there.

## Cheap experiments, by cost

0. Done above: the curves, CPU only.
1. **lambda sweep on the final model's adapter.**
   - First the validation losses (human, robot video, robot action) at 5 values of lambda: minutes each.
   - Then closed loop at lambda 0.5 and 0.75 on starts 8..31.
   - Variants: per layer (LiNeS, ICLR 2025), or scaling the top singular directions of BA (Shuttleworth).
2. **Recovery test** for spurious forgetting: 200-500 steps on human video only. If the loss returns to ~0.25 quickly,
   the drift is alignment, not lost knowledge.
3. **OOD probe:** 1-2 scene perturbations, evaluated on 20000, the best lambda and 12000.
4. **Replay sweep** {0, 5, 15, 30} %, matched on robot updates. Training cost.
5. **Distillation from the LoRA-off model** on human video, ||v_theta - v_base||^2 (DreamBooth-style). Compare with
   replay at the same ratio, and with both together (Riemer). Training cost.

