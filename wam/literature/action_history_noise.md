# Why noise on the action history matters -- literature (3 Nov 2026)

Written after the noise / no-noise comparison and the reliance diagnostic (`../EXPERIMENTS.md`, sections 10 and 15).
Citations were checked against arXiv or proceedings pages; venues marked UNVERIFIED were not confirmed. Numbers are as
reported in the papers.

## Our own evidence (RTX 5090)

- Same data and settings, one seed each, starts 8..31: noise 16/24, no noise 5/24 (paired 13 vs 2, p = 0.007).
  No-noise failures: 18 of 19 are pot-grasp misses; the door (fixed position) opened in 21/24.
- reliance_va.py, teacher-forced, 3 held-out demos. Predicted-motion shift when the cached action
  history is swapped vs when the cached video is swapped:
  - no noise, 14000: 35 / 21 / 34 cm vs 9 / 8 / 10 cm (ratio 2.0-3.0);
  - noise, 14000: 25 / 15 / 26 cm vs 21 / 17 / 21 cm (ratio 0.94-1.10);
  - noise, continued to 20000: ratio 0.74-0.91.
  All three resolve the retry-vs-handle conflict from the scene alike.
- The noise is moderate: (1-s)a + s*eps, with s drawn per frame in [0.001, 0.5] (mean 0.25), in half of the
  samples. The video history gets s in [0.005, 0.83] (mean 0.56).
- LingBot-VA's own stated reason for the noisy video history is speed. It allows partial denoising at test
  time. This is our reading of arXiv 2601.21998, Eq. 10. The paper keeps the action history
  clean and gives no reason for that.

## Copycat / causal confusion (closest mechanism)

- **de Haan, Jayaraman, Levine.** *Causal Confusion in Imitation Learning*. NeurIPS 2019. arXiv 1905.11979.
  With the previous action in the input, the policy needs many more demos or fails, while held-out error
  stays near zero.
- **Wen, Lin, Darrell, Jayaraman, Gao.** *Fighting Copycat Agents in BC from Observation Histories*. NeurIPS
  2020. arXiv 2010.14876. The policy copies the previous action, and it gets worse with temporal
  correlation. Diagnostic: how predictable actions are from past actions.
- **Wen et al.** *Keyframe-Focused Visual Imitation*. ICML 2021. arXiv 2106.06452.
- **Wen et al.** *Fighting Fire with Fire (priming)*. ICML 2022. arXiv 2206.10816. A targeted remedy beats
  generic augmentation.
- **Chuang, Yang, Wen, Gao.** *Residual Action Prediction for copycat*. ECCV 2022. arXiv 2207.09705.
  CARLA: BC-OH 67.1 vs theirs 78.1.
- **Codevilla et al.** *Exploring the Limitations of BC for Driving* ("inertia"). ICCV 2019. arXiv 1904.08980.
- **Bansal, Krizhevsky, Ogale.** *ChauffeurNet*. RSS 2019. arXiv 1812.03079.
  - **Past-motion dropout in 50 % of examples**, because the model could "extrapolate from the past".
  - At closed loop the history is the model's own predictions.
  - Same remedy and the same p as ours; no ablation.
- **Spencer et al.** *Feedback in Imitation Learning: Three Regimes of Covariate Shift*. arXiv 2102.02872.
  "Latching": copycat and covariate shift are one phenomenon under feedback.
- **Swamy et al.** *Causal IL under Temporally Correlated Noise*. ICML 2022. arXiv 2202.01312. Only related.
- **Cheng et al.** *PlanTF*. arXiv 2309.10443 (venue UNVERIFIED). Ego-state dropout 0.75 is the best
  closed-loop setting.
- **Li et al.** *Is Ego Status All You Need...?*. CVPR 2024. arXiv 2312.03031.
- **Counter-evidence: Torne, Tang, Liu, Finn.** *Long-Context Diffusion Policies via Past-Token Prediction*.
  CoRL 2025. arXiv 2505.09561. Diffusion policies *under*-use past actions; copycat is not universal.

## Noise on the conditioning history (exposure bias / drift)

- **Bengio et al.** *Scheduled Sampling*. NeurIPS 2015. arXiv 1506.03099.
- **Ross et al.** *DAgger*. AISTATS 2011. arXiv 1011.0686.
- **Laskey et al.** *DART*. CoRL 2017. arXiv 1703.09327.
- **Ho et al.** *Cascaded Diffusion Models*. JMLR 2022. arXiv 2106.15282. Conditioning augmentation:
  FID 2.35 vs 6.02 without it.
- **Valevski et al.** *GameNGen*. ICLR 2025. arXiv 2408.14837.
  - Noise on the context frames (not on past actions).
  - Without it, quality degrades after 10-20 frames.
- **Chen et al.** *Diffusion Forcing*. NeurIPS 2024. arXiv 2407.01392. An independent noise level per token.
- **Song et al.** *History-Guided Video Diffusion*. ICML 2025. arXiv 2502.06764.
- **Huang et al.** *Self Forcing*. NeurIPS 2025. arXiv 2506.08009.
  - Noisy context alone did not beat clean context: VBench 83.66 vs 84.34; Self Forcing scored 85.07.
  - Noise is not a reliable cure for exposure bias by itself.
- **Alonso et al.** *DIAMOND*. NeurIPS 2024. arXiv 2405.12399. No context noise found.

## Robot policies / world-action models

- **Octo.** RSS 2024. arXiv 2405.12213. Proprio inputs "seemed generally worse", hypothesised as causal
  confusion.
- **Zhao et al.** *Do You Need Proprioceptive States in Visuomotor Policies?*. arXiv 2509.18644.
  - Removing the state leaves in-domain success unchanged.
  - Under spatial shift it helps hugely (π0 height 0 -> 0.984).
  - Noising the state helps less than removing it.
- **Hu et al.** *AR-VLA*. arXiv 2603.10126.
  - Masks the action and proprio history.
  - Mask-rate sweep on SIMPLER WidowX: 0.0 -> 0 %, 0.2 -> 27 %, 0.4 -> 48 %, **0.6 -> 61.5 %**, 0.8 -> 49 %,
    1.0 -> 28 % (an inverted U).
  - Validation error is lowest at mask 0.
- **Li, Gao, Sadigh, Song.** *UVA*. RSS 2025. arXiv 2503.00200. Video supervision "reduces overfitting to
  action history" (figure only).
- **Zhu et al.** *UWM*. RSS 2025. arXiv 2504.02792. Independent timesteps on the targets. Only related.
- **Ye et al.** *DreamZero*. arXiv 2602.15922. Autoregression on video only, "to avoid error propagation
  from closed-loop action prediction".
- **Reflective VLA.** arXiv 2606.25215. History "can induce a shortcut where the policy extrapolates from
  past action chunks".
- **π0** (arXiv 2410.24164) uses no past actions. **OpenVLA** (arXiv 2406.09246) uses no history or
  proprio. Venues UNVERIFIED.

## Reading

| Hypothesis | Support |
|---|---|
| (a) Copycat | Best supported. The trigger is smooth, correlated demos. The signature is a failure only where the target moves (the pot, not the door). ChauffeurNet's p = 0.5 and AR-VLA's sweep match the dose, and our reliance diagnostic measures the shift directly. |
| (b) Exposure bias | Present by construction; not separated from (a). Spencer et al. treat them as one phenomenon. |
| (c) Generic regularisation | Least supported. Targeted remedies beat generic ones in the literature. |
| Seed variance | Not excluded by the one-seed comparison. Needs a second seed. |

## Next diagnostics

1. Dose response: p in {0, 0.25, 0.5, 1 = drop the action history} and the s range. An inverted U, as in
   AR-VLA, would mean the history is useful but must be de-weighted.
2. A second seed for both arms.
3. An equal-strength regulariser that leaves the action history alone, to test (c).
4. Closed loop with the matching demo's clean history vs the model's own executed history, to test (b).
5. Grasp point vs pot position across starts (slope near 0 = replaying a memorised path).
