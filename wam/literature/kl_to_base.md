# Regularising fine-tuning toward the base (KL, distillation, prior preservation) -- literature (4 Nov 2026)

This note follows forgetting.md and does not repeat it: RL's Razor, LwF, WiSE-FT, RETAIN and KI are
covered there. The question here is whether to add beta * ||v_theta(x_t, t) - v_base(x_t, t)||^2 (v_base = the same
model with the LoRA adapter off) to LingBot-VA fine-tuning, on (a) robot training clips or (b) human / generic video
with no targets, and how.

How the numbers were read:
- 73 papers were downloaded from arXiv and read in the full text (pdftotext; arXiv HTML where tables were garbled).
- No number comes from a summarising fetch, so nothing is marked [ext].
- fig-only: the result is shown only in a figure.
- "layout reconstructed": the pdftotext table layout was re-assembled by hand; spot-check before quoting.
- The key numbers (Riemer, Masip, DER, ConSFT, DDRL, Flow-DPO, DPOK, Zajac, Adjoint Matching, Custom Diffusion,
  DreamBooth, Bansal, Info-VLA, Video Policy, DreamZero, EVA, ORW-CFM-W2, Flow-GRPO, and all the theory) were re-checked
  against the text.
- Venues come from the PDF header, the arXiv comments line or the venue page; "arXiv" = no venue found.
- Time convention below: tau = noise level (tau = 1 pure noise), x_tau = (1 - tau) x_0 + tau eps. Wan writes the
  reverse convention; only the labels change.

## 1. Theory: what "KL to the base" means for diffusion and flow models

### 1a. SDE diffusion: the Girsanov claim is right, with four qualifications

- **The exact result.** Song, Durkan, Murray, Ermon (NeurIPS 2021, 2101.09258), Thm 1 and App. eq. 17-18.
  - Take two reverse SDEs with the same diffusion coefficient g(t), the same starting point, and drifts f - g^2 s.
  - Then KL(mu || nu) = 1/2 * int_0^T E_{p_t} [ g(t)^2 ||s_1(x,t) - s_2(x,t)||^2 ] dt = J_SM(theta; g^2).
  - Written in drifts: 1/2 int E ||b_1 - b_2||^2 / g(t)^2 dt.
  - The score weight lambda(t) = g(t)^2 is the "likelihood weighting".
  - With a prior mismatch: KL(p || p_theta^SDE) <= J_SM(theta; g^2) + KL(p_T || pi). The data-processing inequality
    takes it from paths to the final sample.
- **Same identity in the other frameworks:**
  - Albergo, Boffi, Vanden-Eijnden (JMLR 26, 2025, 2303.08797), Lemma 22: for FPEs with diffusion eps > 0,
    KL(rho(1) || rho_hat(1)) <= 1/(4 eps) int int |b_hat_F - b_F|^2 rho dx dt.
  - Adjoint Matching (Domingo-Enrich et al., ICLR 2025, 2409.08861), eq. 18: the control cost E int 1/2 ||u||^2 dt is
    the path KL, with b_ft = b_base + sigma u.
  - Chen et al. (ICLR 2023, 2209.11215) make Girsanov rigorous without Novikov's condition. The approximation argument
    yields an inequality, not an equality.
- **Qualifications you need (corrections to the one-line claim):**
  1. **Factor and weighting.** The KL is 1/2 * (drift difference)^2 / g^2, equivalently g^2/2 * (score difference)^2.
     It is not an unweighted integral.
  2. **Expectation.** The expectation is over the paths of the first argument. For KL(P_theta || P_base) that means the
     fine-tuned model's own sampled trajectories, not data noised by the forward process. DPOK, Flow-GRPO and Adjoint
     Matching all compute it on their own rollouts. Diffusion-DPO replaces the reverse process with the forward q, which
     is an approximation.
  3. **Same diffusion and same start.** Both SDEs need the same g(t) and initial law. If g differs, the path measures
     are mutually singular and the KL is infinite. A different start adds KL(p_T || pi).
  4. **It bounds the KL of the outputs from above, not exactly.** Path KL >= KL of the final samples.
     - Adjoint Matching shows the naive KL-regularised objective is biased by the initial value function V(X_0, 0):
       "when the sampling procedure is noiseless, i.e., sigma(t) = 0, fine-tuning naively will not have any effect"
       (L420-426).
     - The fix is the "memoryless" noise schedule.
     - This matters for reward fine-tuning through a sampler, not for supervised flow matching like ours.

### 1b. Deterministic ODE flow matching: ||v - v_base||^2 does not bound KL by itself

- **The path KL is undefined (infinite)** for two different deterministic flows. Only the KL of the final
  distributions is meaningful.
- **Exact KL needs the score of the model's own density, not just the velocity:**
  - Albergo et al. (JMLR 2025), Lemma 21: KL(rho(1) || rho_hat(1)) = int int (grad log rho_hat - grad log rho) .
    (b_hat - b) rho dx dt.
  - "it is insufficient in general to match b_hat with b to obtain control on the KL"; you also need the Fisher
    divergence.
  - Lu et al. (ICML 2022, 2206.08265), Thm 3.1: KL(q_0 || p_0^ODE) = KL(q_T || p_T^ODE) + J_SM + J_Diff. J_Diff involves
    grad log p_t^ODE - s_theta, and score matching "cannot control" it. They bound it with J_ODE <= sqrt(J_SM * J_Fisher).
- **Bounds that do exist (all need regularity of the learned field):**
  - **W2** (Benton, Deligiannidis, Doucet; TMLR 2024, 2305.16860), Thm 1: W2(pi_hat_1, pi_1) <= eps * exp(int_0^1 L_t dt).
    - eps^2 = int E_{X_t} ||v_theta - v^X||^2 dt, with the expectation under the reference process's marginals.
    - L_t is the spatial Lipschitz constant of v_theta.
    - Albergo & Vanden-Eijnden (ICLR 2023, 2209.15571), Prop 3, is the earlier version with a uniform constant:
      W2^2 <= e^{1+2K} H(v_hat), where H = int int |v_hat - v|^2 rho_t.
  - **W2 between two flow models.** ORW-CFM-W2 (Fan et al., ICLR 2025, 2502.06061), Thm 3:
    - W2^2(p_1^{theta1}, p_1^{theta2}) <= e^{2L} int E_{x ~ p_s^{theta1}} ||v^{theta1} - v^{theta2}||^2 ds, with L the
      Lipschitz constant of v^{theta2}.
    - **This is the published justification of our exact candidate term**, and they use it as a regulariser with
      alpha = 1.
  - **KL** (Su et al., 2511.05480, arXiv), Thm 3.1: KL(p_1 || q_1) <= A_1 eps + A_2 eps^2.
    - A_1 and A_2 are exponential in bounds on ||grad v||, |div u - div v|, ||grad div v|| and the data score.
    - So KL scales like sqrt(regulariser), not linearly, and only under sup-norm regularity.
  - **TV with a deterministic sampler** (Li et al., ICLR 2024, 2306.09251) also needs the mean Jacobian error of the
    score, not just L2. Benton et al. (TMLR) say the same of that line of work.
- **Verdict.** ||v - v_base||^2 is a proper surrogate for W2, up to the exp(Lipschitz) factor. It is a surrogate for KL
  only with extra regularity, or once a stochastic sampler is chosen (1c). It is not "the KL".

### 1c. The KL weight a flow model actually implies (derived here from Flow-GRPO's equations)

- **Flow-GRPO (NeurIPS 2025, 2505.05470), eq. 8-9 and the closed form after them:**
  - Rectified flow has a family of SDEs with the same marginals: dx = [v + sigma_tau^2/(2 tau) (x + (1 - tau) v)] dtau
    + sigma_tau dw.
  - The per-step KL is closed form: D_KL = (Delta tau / 2) (sigma_tau (1 - tau)/(2 tau) + 1/sigma_tau)^2
    ||v_theta - v_ref||^2.
- **Derivation (ours, from that formula):**
  - Minimising the weight over sigma gives w*(tau) = (1 - tau)/tau, at sigma_tau^2 = 2 tau/(1 - tau).
  - Flow-GRPO's own sigma_tau = a sqrt(tau/(1 - tau)), a = 0.7, gives (1/a + a/2)^2/2 * (1 - tau)/tau = 1.58 (1 - tau)/tau.
  - So the tightest Girsanov weight on ||Delta v||^2 is large at low noise, infinite as tau -> 0, and 0 at pure noise.
- **Consequences:**
  - The usual unweighted term over the training tau distribution is not a KL. It over-weights high noise relative to
    any path KL.
  - The derived weighting is the one that failed in practice (Flow-DPO, Section 4). It is not a reason to reweight
    toward low noise.

### 1d. Two identities for our case (derived, exact)

- **D1: on the task inputs, the term only shrinks the target.**
  - The pointwise minimiser of E||v - u||^2 + beta ||v - v_base||^2 is v* = (E[u | x_tau, c] + beta v_base)/(1 + beta).
  - Option (a) is therefore SFT toward a blend of the target and the base. On robot clips it leaves a fraction
    beta/(1 + beta) of the data-to-base gap unlearned (9 % at beta = 0.1).
  - The blend is not the velocity of any mixture distribution, which would need x-dependent weights.
  - Unlike reward-RL with KL (RL's Razor picks the nearest of many reward-optimal solutions), fixed regression targets
    leave no "nearest solution" to pick. On-task KL can only learn less.
- **D2: the human validation loss is almost the regulariser.**
  - Let v*_H = E[u | x_tau, c] on human clips and D_H(theta) = E_H ||v_theta - v_base||^2. Then
    L_H(theta) - L_H(base) = D_H(theta) + 2 E_H <v_theta - v_base, v_base - v*_H>.
  - If the base is near-optimal on human video, the rise 0.240 -> 0.410 is about D_H.
  - Two corollaries:
    - D_H can be measured now on the saved checkpoints.
    - Option (b) on the same human dataset optimises nearly the metric itself, so held-out human loss stops being an
      independent forgetting measure.

## 2. Consensus (several independent groups)

1. **On SDEs the KL to the base is an integrated, weighted squared drift difference along the model's own paths. On ODEs
   velocity L2 controls only W2, plus KL under regularity.**
   - Song 2021; Chen 2023; Albergo JMLR 2025; Domingo-Enrich 2025; Lu 2022; Benton 2024; Li 2024; Su 2025.
2. **Anchoring only on the new-task inputs protects little when those inputs are far from the old ones, and it costs
   task learning.**
   - DER (NeurIPS 2020, 2004.07211): LwF, distillation on new-task inputs, scores 19.61 on S-CIFAR-10 class-IL vs SGD
     19.62.
   - Masip et al. (CoLLAs 2024, 2311.14028), distilling on current-task data: FID 65.01 vs finetune 68.76 on
     Fashion-MNIST, and 62.09 vs 55.77 on CIFAR-10, worse than finetune.
   - **ConSFT** (2605.08879, arXiv) on pi0, the only test of our exact term in a flow VLA: velocity MSE to the frozen base
     on the target-task inputs, lambda = 0.01.
     - Target 0.80 vs 0.90 for the others; prior suites 0.05 vs 0.09 for plain SFT.
     - It "restricts target-task convergence" (RoboTwin 0.45 vs SFT 0.55).
   - DPOK's KL-O (the same term on fixed SFT samples) "generally suffers from lower ImageReward" (fig-only).
   - Counter-example: Riemer et al. (LLMs, KL on the fine-tuning inputs) works, but there the text inputs overlap the
     old distribution (see Contested).
3. **On the same old or generic inputs, matching the base's outputs is at least as good as ground-truth replay. Doing
   both is best.**
   - DER, same buffer of 500: ER 57.74, DER 70.51, DER++ 72.70.
     - The gap shrinks with a large buffer: at 5120, ER 82.47, DER 83.81, DER++ 85.24.
   - Masip, on the same generated inputs: generative replay (true-noise target) FID 211.79 vs generative distillation
     (old model's eps) 23.53. That is close to real-data replay with 1000 samples (21.6).
   - Marek et al. (2605.26097, arXiv; fig-only):
     - KL on replay data beats next-token loss on the same data.
     - Ground-truth loss on substitute (non-pretraining) data "actually increases forgetting compared to standard
       finetuning". KL on that data does not.
   - Riemer: replay plus KL is best.
   - Info-VLA (2603.13335, arXiv; pi0.5, feature distillation on replay samples): AA 73.3 vs ER 67.2.
   - None of these is a video-diffusion or flow-VLA test.
4. **A KL or base anchor trades task gain for quality or diversity, and never fully prevents diversity loss or
   hacking.** "KL alone learns less", with numbers:
   - Riemer, plasticity (task gain) vs plain fine-tuning at 2.5: KL beta = 0.1 gives 0.7 (28 %), beta = 0.01 gives 1.8
     (72 %), beta = 0.001 gives 2.7 but forgetting 8.4 vs 15.4.
   - DDRL (2512.04332, arXiv), Cosmos2.5-2B text-to-video: Flow-GRPO with KL (beta = 0.01, swept over [0.001, 0.1])
     reward 0.408 = the base's 0.408; DanceGRPO without KL reached 0.715 but hacked.
   - Perceptual entropy (2605.12112, arXiv), FLUX: PickScore 0.226 base, 0.243 no KL, 0.233 with KL 0.001, i.e. about 41 %
     of the gain. DINO diversity: 104.76 base, 84.17 no KL, 101.11 with KL.
   - Adjoint Matching: DreamSim diversity falls from 56.53 (base) to 43.10 (lambda = 1000) and 37.24 (lambda = 12500).
     DRaFT-1 falls to 27.39.
   - Flow-GRPO: same final reward with KL, but slower (fig-only).
   - Diffusion-DPO: a large beta "greatly restricts any appreciable adaptation".
5. **Full-weight replay or retention terms overfit a small retention set, mostly at low noise.**
   - Zajac et al. (2303.15342, arXiv), MNIST:
     - Experience replay with coefficient 1: FID-forgetting 10.62.
     - With coefficient 0.01: 1.00.
     - Finetune: 55.59.
   - Overfitting to the buffer shows at tau of about 50/1000, not 700 (fig-only).
   - Custom Diffusion (CVPR 2023, 2212.04488) with generated regularisation images: KID 34.70, worse than no
     regularisation (32.64). Real images give 20.96.

## 3. Contested or single-paper

- **KL on the task inputs alone (option a).**
  - Riemer et al. (2512.22337, arXiv; LoRA LLMs), on the fine-tuning inputs:

    | Setting | Forgetting | Plasticity |
    |---|---|---|
    | Plain fine-tuning | 15.4 | 2.5 |
    | KL beta = 0.01 | 1.2 | 1.8 |
    | Approximate replay (OpenWebText), 1x | 5.4 | 2.4 |
    | Replay 1x + beta = 0.01 | -0.1 | 1.5 |
    | Replay 3x + beta = 0.001 | 2.0 | 2.7 |

  - So it works when task and old inputs overlap: chat text vs web text, or personalisation prompts (DCO, NeurIPS
    2024, 2402.12004).
  - Under a strong input shift it fails (point 2 of the Consensus). Sim kitchen vs egocentric human video is a strong
    shift.
- **Which noise levels to protect.**
  - Flow-DPO / VideoAlign (NeurIPS 2025, 2501.13918):
    - The derived KL schedule beta_t = beta (1 - t)^2, t = noise, is weak at high noise. It "degrades alignment ...
      leading to reward hacking and visual artifacts".
    - Constant beta wins: VBench total 83.41 vs 80.90 for the derived schedule; pretrained 83.19.
  - Zajac: fine-tuning forgetting appears in the high-noise loss (t = 700); the low-noise loss (t = 50) barely moves
    (fig-only).
  - CustomCrafter (AAAI 2025, 2408.13239): LoRA scale 0.4 for the first 5 of 50 (high-noise) steps, then 0.8, to keep
    the video model's motion.
  - DPPO (ICLR 2025, 2409.00588): keeps the base frozen for the early (high-noise) denoising steps and fine-tunes only
    the last 10 of 100. The reason is efficiency.
  - Against: Diff-Tuning (NeurIPS 2024, 2406.00773) retains low noise (xi(t) = 1 - t^tau) and adapts at high noise. It
    is built for transfer and never measures retention.
  - Net: nobody supports dropping the high-noise steps from a retention term. Several groups find the prior, i.e.
    motion and layout, sits there.
- **On-policy (reverse) KL vs forward KL / data regularisation.**
  - DDRL argues the reverse KL on the model's own samples is "adversarially hacked": x_t becomes out-of-distribution
    for the reference, and at beta = 0.1 the KL stays flat while the videos get noise textures.
  - Its fix is the diffusion loss on reference data, i.e. replay. One group (NVIDIA), about 1M GPU-hours.
  - DRIFT (NeurIPS 2026, 2601.12401): "push-back signal only for samples generated by pi_theta".
- **Real vs generated retention inputs.**
  - Generated inputs did worse in Custom Diffusion (KID above) and C-LoRA (TMLR 2024, 2304.06027: generative replay
    worst, A_mmd 17.34 vs 5.57 for sequential LoRA).
  - They did better in Masip (with distillation targets), Bansal & Sanghavi (2505.13811, arXiv; self-generated text
    keeps the pre-existing average at 28.24 vs base 29.50, pretraining replay 24.73, FT 23.55) and DDRL.
  - Reading: generated inputs are fine with teacher targets and poor with "ground-truth" targets on bad samples.
- **A KL term can raise the task reward in online RL** (it prevents collapse). ORW-CFM-W2, SD3, one prompt, 64 samples:
  - With W2, CLIP 35.46 and diversity 4.21.
  - Without, CLIP 33.63 and diversity 3.47.
  - Base: CLIP 28.69 and diversity 4.78.
  - This does not transfer to SFT with fixed targets (D1).
- **Robot video world models.** No paper measures forgetting of the video prior together with a KL or distillation
  anchor.
  - The retention devices used:
    - LoRA, asserted only (DreamGen, CoRL 2025: rank 4 "to mitigate forgetting").
    - Freezing the video model after a video stage. Video Policy (2508.00795, arXiv): joint training 0.57 vs 2-stage
      0.63 vs no video tuning 0.09.
    - 1:1 co-training with pretraining data (DreamZero, 2602.15922, arXiv: 38.3 % -> 55.4 % / 54.3 % on unseen tasks,
      no ablation of the mix).
    - KL only inside GRPO (EVA, 2603.17808, arXiv: beta = 0.004, still hacks and needs early stopping; never ablated).

## 4. Paper table

| Citation | Claim | Numbers read | Relevance |
|---|---|---|---|
| Song et al., NeurIPS 2021, 2101.09258 | Path KL via Girsanov = J_SM with lambda = g^2 | Thm 1, eq. 17-18: 1/2 int E g^2 \|\|s - s'\|\|^2 | The exact Girsanov form; expectation under the first path measure |
| Chen et al., ICLR 2023, 2209.11215 | Girsanov without Novikov (approximation argument) | -- | Rigour of the SDE claim |
| Benton, De Bortoli, Doucet, Deligiannidis, ICLR 2024, 2308.03686 | KL(q_delta \|\| p) <~ eps_score^2 + kappa^2 dN + kappa dT + d e^{-2T} | Thm 1 | KL is linear in score MSE for SDE sampling |
| Lu et al., ICML 2022, 2206.08265 | ODE KL = J_SM + J_Diff; J_Diff not controlled by score matching | Thm 3.1, eq. 11 | ODE needs more than L2 |
| Albergo, Boffi, Vanden-Eijnden, JMLR 2025, 2303.08797 | Transport-equation KL needs Fisher; FPE KL <= 1/(4 eps) int \|\|Delta b\|\|^2 | Lemmas 21-22, Thm 23 | Core theory |
| Albergo & Vanden-Eijnden, ICLR 2023, 2209.15571 | W2^2 <= e^{1+2K} int \|\|v_hat - v\|\|^2 | Prop 3 | W2 surrogate |
| Benton, Deligiannidis, Doucet, TMLR 2024, 2305.16860 | W2 <= eps exp(int L_t) for deterministic FM | Thm 1 | W2 surrogate with a time-varying Lipschitz constant |
| Su et al., 2511.05480, arXiv | Deterministic FM: KL <= A_1 eps + A_2 eps^2 | Thm 3.1 | KL ~ sqrt(regulariser) |
| Li et al., ICLR 2024, 2306.09251 | Deterministic sampler TV needs Jacobian error | Assumption 2 | Same |
| Flow-GRPO, NeurIPS 2025, 2505.05470 | Per-step KL = Delta t/2 (sigma(1-t)/(2t) + 1/sigma)^2 \|\|Delta v\|\|^2; KL stops hacking | beta 0.04 (GenEval, OCR), 0.01 (PickScore); a = 0.7; T = 10 train; GenEval task no KL vs KL: Aesthetic 4.93 vs 5.25, DeQA 2.77 vs 4.01, ImageReward 0.44 vs 1.03 (HTML table) | The velocity-KL formula; beta scale |
| ORW-CFM-W2, ICLR 2025, 2502.06061 | W2 bound gives the regulariser \|\|v_ft - v_ref\|\|^2 on own samples | alpha = 1; SD3 one prompt: CLIP / diversity 28.69 / 4.78 base, 33.63 / 3.47 no W2, 35.46 / 4.21 with W2 | Our exact term, on-policy, in RL |
| Adjoint Matching, ICLR 2025, 2409.08861 | KL = control cost; memoryless schedule removes the initial value bias | Diversity 56.53 base -> 43.10 (lambda 1000), 37.24 (lambda 12500); DRaFT-1 27.39 | KL does not stop diversity loss |
| DPOK, NeurIPS 2023, 2305.16381 | Sum of per-step Gaussian KLs; KL-O = our term on SFT samples | alpha = 10, beta = 0.01; KL-O gamma = 2 from {0.1, 1, 2, 5}; KL-O lowers ImageReward (fig-only) | Option (a) in SFT loses reward |
| DDPO, ICLR 2024, 2305.13301 | No KL; early-stops by hand | -- | Negative control |
| ImageReward / ReFL, NeurIPS 2023, 2304.05977 | Regularises with the pretraining loss (replay), not KL | 64 + 64 per batch of 128; reward only at t in [1, 10] of 40; no ablation | Replay as the anchor |
| DRaFT, ICLR 2024, 2309.17400 | LoRA weight decay toward pretrained "slightly improve"; LoRA scaling beats a KL on eps at the last step | fig-only | Weight-space alternative |
| Diffusion-DPO, CVPR 2024, 2311.12908 | Implicit KL; beta in [2000, 5000] | SD1.5 2000, SDXL 5000; large beta "restricts any appreciable adaptation" | beta has no shared scale |
| Flow-DPO / VideoAlign, NeurIPS 2025, 2501.13918 | Derived beta_t = beta(1-t)^2 hurts; constant beta | VBench total 83.41 vs 80.90 (derived) vs 83.19 (pretrained); beta 500, LoRA r64 | Timestep weighting on video RF |
| DDRL, 2512.04332, arXiv | Reverse KL on own samples is unreliable; forward KL = replay | Cosmos2.5-2B reward: base 0.408, DanceGRPO 0.715 (hacked), Flow-GRPO+KL 0.408, DDRL 0.604; on-policy KL about 2x NFEs | Against on-policy KL |
| Perceptual entropy, 2605.12112, arXiv | KL keeps diversity but cuts the gain | PickScore 0.226 / 0.243 / 0.233; DINO diversity 104.76 / 84.17 / 101.11 | "KL learns less" |
| DRIFT, NeurIPS 2026, 2601.12401 | KL pushes back only on own samples | fig-only | Same |
| EVA, 2603.17808, arXiv | GRPO on a video world model, KL beta 0.004, LoRA r32 | Still hacks; early-stopped | Robot world model, KL not ablated |
| DreamBooth, CVPR 2023, 2208.12242 | Prior loss = replay on base-generated samples, lambda = 1 | PRES 0.493 vs 0.664, DIV 0.391 vs 0.371, DINO 0.684 vs 0.712 (with vs without) | The prior-preservation baseline |
| Custom Diffusion, CVPR 2023, 2212.04488 | Real regularisation set; generated ones overfit | KID 20.96 real, 32.64 none, 34.70 generated | Use real retention inputs |
| C-LoRA, TMLR 2024, 2304.06027 | Weight-space regulariser beats generative replay | A_mmd 2.04 vs 5.57 (LoRA) vs 17.34 (generative replay) | LoRA sequences forget |
| Zajac et al., 2303.15342, arXiv | Forgetting at high noise; replay weight 0.01 best | FID-forgetting 55.59 FT, 13.11 L2, 10.62 ER x1, 1.00 ER x0.01 | Per-timestep diagnostic; replay weight |
| Masip et al., CoLLAs 2024, 2311.14028 | Teacher-eps distillation on generated inputs >> generative replay; current-data distillation fails | F-MNIST FID 23.53 vs 211.79 vs 65.01 (current data) vs 38.87 (noise inputs) vs 21.6 (real replay, 1000); lambda 0.75; 1.13x wall time | The closest diffusion evidence for option (b) |
| Diff-Tuning, NeurIPS 2024, 2406.00773 | Retain low t, adapt high t | Target FID 14.36 -> 10.63 (avg 8 tasks); no retention metric | Opposite timestep choice, untested for retention |
| DCO, NeurIPS 2024, 2402.12004 | Implicit KL on new-task images only | beta_t = 1000; fig-only | Option (a) where inputs overlap |
| CustomCrafter, AAAI 2025, 2408.13239 | Lower LoRA scale on high-noise steps keeps motion | 0.4 for 5 of 50 steps, then 0.8 | Video prior lives at high noise |
| Still-Moving, ACM TOG 2024, 2407.08674 | 40 prior-preservation videos | qualitative only | Video prior preservation |
| DER / DER++, NeurIPS 2020, 2004.07211 | Logit replay > label replay; both best | Buffer 500: 57.74 / 70.51 / 72.70; buffer 5120: 82.47 / 83.81 / 85.24; LwF 19.61 | Distill vs replay at equal ratio |
| Riemer et al., 2512.22337, arXiv | KL on fine-tuning inputs; plus replay best | Table in Section 3; KL is 1.5x compute, less with LoRA | "KL learns less" numbers |
| Marek et al., 2605.26097, arXiv | KL on replay > NTP on replay; NTP on substitute data hurts | fig-only; lambda = 10 | Ground-truth replay on a non-pretraining set can hurt |
| Bansal & Sanghavi, 2505.13811, arXiv | Base-sampled data beats real pretraining replay | Pre-existing avg / GSM8K: base 29.50 / 1.59, FT 23.55 / 29.49, pretraining replay 24.73 / 15.09, self-generated 28.24 / 29.34 | Forward KL by sampling |
| ConSFT, 2605.08879, arXiv | Velocity-MSE LwF on target inputs fails in pi0 | LIBERO target / prior: SFT 0.90 / 0.09, LwF 0.80 / 0.05, ER (1:1) 0.90 / 0.18, LoRA r16 0.90 / 0.24; lambda 0.01, no sweep | Our term, option (a), flow VLA |
| FQL, ICML 2025, 2502.02538 | Distil a 1-step policy to the BC flow on dataset states with fresh noise, no targets; upper-bounds W2 | alpha tuned per task | Target-free flow distillation exists |
| DPPO, ICLR 2025, 2409.00588 | No regulariser; base frozen for early steps | Fine-tunes the last 10 of 100 | Step-restricted adaptation |
| ReinFlow, NeurIPS 2025, 2505.22094 | Optional W2 to the pretrained policy; entropy works better | fig-only | -- |
| FDPP, 2501.08259, arXiv | DPOK-style KL on diffusion policy; small KL costs success | alpha sweep fig-only | Option (a) costs task |
| DSRL, CoRL 2025, 2506.15799 | Steer latent noise, weights frozen | Retention not measured | Zero-forgetting alternative |
| Info-VLA, 2603.13335, arXiv | Replay + feature distillation (pi0.5) | AA 73.3 vs ER 67.2 vs sequential 42.1 | Replay + distillation > replay in a VLA |
| DreamGen, CoRL 2025, 2505.12705 | LoRA r4 "to mitigate forgetting" | Not measured | Assertion only |
| DreamZero, 2602.15922, arXiv | 1:1 co-training with pretraining data | 38.3 -> 55.4 / 54.3 % | Co-training in a joint world-action model |
| Video Policy, 2508.00795, arXiv | Freezing the video model after video tuning > joint | 0.63 vs 0.57 vs 0.09 | Action gradients into shared video weights cost |
| Liu et al., ICML 2026, 2603.03818 | Pretrained VLAs need little replay | "buffer size is 2 % ... Pi0 and GR00T still have low NBT around 0.1-0.2"; pi0 LIBERO-Object NBT sequential 0.696 vs ER -0.007 (layout reconstructed) | Small replay largely suffices in VLA continual learning |

## 5. What applies to our setup (reading of the above)

- **Use option (b), not (a).**
  - (a) on robot clips is SFT toward a shrunk target (D1). It constrains outputs exactly where they must change.
  - It is the case that failed in ConSFT (flow VLA), Masip (current-data distillation) and DER (LwF).
  - On robot clips, "LoRA off" is not the base either: the fully trained action embedder and projection feed action
    tokens the base backbone never saw. That gives a reference evaluated out of its distribution, DDRL's failure mode.
  - (b) on human / generic video involves no action tokens, so the LoRA-off model is the true base there.
- **Apply the term to the video velocity only.** The base's action outputs mean nothing for our action interface.
  In a mixture-of-transformers WAM such as Fast-WAM, one can anchor only the 5B video expert's velocity on human video and
  leave the action expert free.
- **What the term is in theory.**
  - In (b), the expectation is over forward-noised human clips: the base's own data marginals, which is what Benton's
    W2 bound wants.
  - With the training tau distribution and no weight it is a W2-type surrogate, not a KL (1b, 1c).
  - That is fine. The literature does not support switching to the "KL-correct" weighting.
- **Timesteps.** Use all, with the training tau sampler and constant weight.
  - Flow-DPO: constant beats the derived low-noise-heavy schedule.
  - Zajac and CustomCrafter: the prior lives at high noise.
  - Do not drop high-noise steps. Do not adopt (1 - tau)/tau.
- **beta.** No value transfers across papers: Flow-GRPO 0.01-0.04, which others cite as 0.001-0.004; Diffusion-DPO
  2000-5000; Flow-DPO 500.
  - In our units (velocity MSE, the same as the training loss), the precedents for a supervised retention term are
    about 1: DreamBooth lambda = 1, ORW-CFM-W2 alpha = 1, Masip lambda = 0.75, DER++ alpha = beta = 0.5.
  - Start at beta = 1 on the human slice, and sweep {0.3, 1, 3}.
  - Use Flow-GRPO's rule: monitor D_H and keep it small and roughly flat.
  - For scale: the robot-only drift is about +0.17 in human loss at 20000, against a robot video loss of about 0.07.
- **Distillation or ground-truth replay at the same 30 %.**
  - The evidence (DER, Masip, Marek; not video) says distillation is at least as good, and both is best (DER++,
    Riemer, Info-VLA).
  - Our 30 % co-training run argues for distillation in particular. Its human loss fell to 0.152, below the base's
    0.240, so ground-truth replay on that ego dataset teaches the dataset rather than retaining the base. That is
    Marek's "substitute data" effect.
  - Distillation cannot go below the base and cannot overfit the human takes the way replay can.
  - DER's buffer-5120 result warns that the advantage shrinks when replay data are plentiful, which they are here.
- **Overhead (estimate, not from a paper).** One no-grad forward of the same weights with the adapter off, per
  regularised clip; no extra weight memory with LoRA.
  - On a 30 % slice this is about +10 % FLOPs: a forward is about 1/3 of a forward-backward, less with activation
    checkpointing.
  - Caching v_base for K fixed (tau, eps) draws per human clip removes it entirely, at the cost of fixed noise draws.
  - Riemer reports 1.5x for KL on every sample in full fine-tuning, less with LoRA. Masip reports 1.06-1.13x over
    replay.
- **Pitfalls.**
  - Distilling on the same human dataset the validation takes come from makes the held-out human loss nearly the
    objective (D2). Retention then needs an independent probe: a different video source, generation quality, or a
    robot OOD scene.
  - beta too large freezes learning (Riemer: plasticity 0.7 at beta = 0.1).
  - A KL can look controlled while samples degrade (DDRL).
  - Prefer real retention clips to base-generated ones (Custom Diffusion, C-LoRA).
- **Still unproven.** The point from forgetting.md stands: the task is at 99 %, so retention has value only if it
  buys OOD robustness, and none of this tests that.

## 6. Cheap experiments, by compute

0. **Forward-only, on saved checkpoints (minutes each):**
   - (i) D_H(theta) = E_H ||v_theta - v_base||^2 on held-out human clips at 2000 ... 20000 of the robot-only run and the
     30 % human run. Does D_H track the 0.17 rise (D2)?
   - (ii) The same on robot clips, D_R: how far the model moved where it had to.
   - (iii) The rise of the human loss per tau bin, base vs 20000 (Zajac-style). This tells whether the forgetting sits
     at high noise and whether to bother with timestep variants.
1. **Zero-overhead knobs:** decoupled weight decay toward 0 on the LoRA factors, i.e. toward the base. Zajac's L2 cut
   FID-forgetting 55.59 -> 13.11; DRaFT found it slightly helpful. Use this as one short-run arm.
2. **Precompute v_base** for a fixed human subset (K = 2-4 noise draws per clip). This is one forward pass over the
   subset.
3. **Short runs (6000 steps, where the drift starts), matched on robot updates.** Arms:
   - robot-only;
   - 30 % ground-truth replay;
   - 30 % distil-only, beta = 1;
   - 30 % both (DER++-style 0.5 / 0.5);
   - one arm of option (a) at beta = 0.1, only to confirm D1.

   Read robot validation loss, D_H, and an independent video probe.
4. **beta sweep {0.3, 1, 3}** on the winner, then a high-noise-only variant (tau > 0.5) if step 0(iii) puts the
   forgetting there.
5. **Full 20000 runs** of the best arm vs robot-only, with closed loop on starts 8..31 and the OOD probe. This is the
   only step that shows retention is worth anything.
