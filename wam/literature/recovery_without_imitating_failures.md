# Teaching recovery without teaching the failure -- literature (4 Nov 2026)

Written after the planned-door-miss census (`../EXPERIMENTS.md`, section 17): the policy imitates the scripted
handle-miss demos as planned misses.

The arXiv full texts of the main papers were read (method sections, numbers). For RACER the code was read too, because
the paper text does not say which transitions are trained on. UNVERIFIED means the venue could not be confirmed, or only the
metadata or abstract was read.

## A. Keeping mistakes or perturbations out of the supervised targets, or down-weighting them

- **DART** (Laskey et al., CoRL 2017, arXiv 1703.09327).
  - Failure source: Gaussian noise added to the supervisor's *executed* actions.
  - Supervision: on the states the noisy run visits, the label is the supervisor's *noise-free* action.
  - Reported: HSR grasping in clutter 79 % vs BC 49 %.
- **ChauffeurNet** (Bansal et al., RSS 2019, arXiv 1812.03079).
  - Failure source: synthesized perturbations (position +-0.5 m, heading +-pi/3), plus a smooth trajectory back to the goal.
  - Supervision: the deviation appears only in the input history; the target is the future that returns.
  - Perturbed examples get weight 1/10, "to avoid learning a propensity for perturbed driving". 50 % past-motion dropout.
  - Reported: with no perturbation data (M0) nothing recovers; M3/M4 all recover.
- **HG-DAgger** (Kelly et al., ICRA 2019, arXiv 1810.02890). Expert labels are collected only on the recovery trajectories.
- **EIL** (Spencer et al., RSS 2020).
  - The last fraction before the takeover is labelled "bad" and used as a Q constraint.
  - The intervention actions are the explicit labels.
- **IWR** (Mandlekar et al., arXiv 2012.06733).
  - Robot samples and intervention samples go into separate pools, sampled 50/50.
  - Reported: Coffee 87.5 % vs HG-DAgger 69.6 %.
- **Sirius** (Liu et al., RSS 2023, arXiv 2211.08416).
  - The 15 steps before a takeover are a "preintv" class, weighted 0.
  - The ablation shows that treating them as normal samples hurts. The paper notes BC-RNN "performs poorly due to copying the
    suboptimal behaviors".
- **ThriftyDAgger** (Hoque et al., CoRL 2021). The policy is trained only on the human segments.
- **RLIF** (Luo et al., ICLR 2024). Reward -1 just before an intervention; off-policy RL.
- **CCIL** (Ke et al., ICLR 2024). Synthetic corrective labels from locally Lipschitz dynamics, mainly low-dimensional state.
- **pi\*0.6 RECAP** (Physical Intelligence, arXiv 2511.14759).
  - Everything is supervised, but with an "Advantage: positive/negative" text condition; corrections are forced positive, and
    inference uses positive.
  - The paper criticises AWR-style methods for discarding or heavily down-weighting data.

## B. Generating failure-and-recovery data by perturbation

- **RACER** (Dai et al., ICRA 2025, arXiv 2409.14674).
  - Failure source: noise on expert key-frame actions; only successful rollouts are kept.
  - Supervision: the code (`generate_replay_buffer.py`) writes only expert->expert, failure->intermediate and failure->expert
    transitions, so the step that caused the failure is not trained on.
  - Reported: 70.2 % vs 62.9 % over 18 tasks, but only about 2 % of that is attributed to the recovery data.
- **IntervenGen** (Hoque et al., IROS 2024, arXiv 2405.01472). The closest to our setting.
  - Failure source: the policy runs with noisy object-pose observations; human recoveries are replayed MimicGen-style.
  - Supervision: only the recovery segment xi[t:] is kept, which "can prevent the imitation of mistakes".
  - The failure must be observable at contact, otherwise recovery could not be learned.
  - Reported: Nut Insertion 98 % vs 64 %.
- **JUICER** (Ankile et al., IROS 2024). Random nearby robot poses at bottleneck states; only the way back is stored.
- **FLARE** (Zhao et al., CVPR 2026, arXiv 2608.26645).
  - Failure source: random perturbation motions between sub-tasks.
  - Supervision: "perturbation actions ... are not used as training targets".
  - Reported: pi0.5 backbone 72.2 -> 84.0 %.
- **FailSafe** (Lin et al., arXiv 2510.01642). Injects failures into ManiSkill plans; a separate VLM outputs the recovery action.
- **AHA / FailGen** (Duan et al., ICLR 2025). Perturbed key frames (e.g. No_Grasp, Slip); trains a failure-detection VLM only.
- **PGDG** (arXiv 2605.21710). Physics-grounded recovery trajectories, relabelled by short-horizon control.
- **RaC** (Hu et al., arXiv 2509.07953). Real robot, human rewind and correct; robot rollouts are not kept unless the episode
  succeeded without intervention.
- **Diffusion Policy** (Chi et al., RSS 2023). Perturbations at evaluation only (occluding the camera, moving the T).
- **pi0** (RSS 2025) says high-quality data alone does not teach recovery, and that pretraining gives a repertoire of corrections.

## C. Work that observes the planned-failure effect itself

- **RePO-VLA** (arXiv 2605.09410).
  - Quote: "Imitating that whole episode can teach the prefix as a trigger for correction ... makes the original mistake look
    necessary."
  - Fold Towel, 10 trials each:

    | Training data | Success |
    |---|---|
    | BC on success only | 40 % |
    | raw aggregation, failure prefix kept | 0 % |
    | prefix cut, recovery as its own episode (TSER) | 40 % |
    | with value conditioning | 70 % |

- **VBT** (Brandfonbrener et al., arXiv 2210.02343) is counter-evidence.
  - Deliberate near-miss data with recovery still helped BC: 73 % vs 66 % on success-only data.
  - It did not measure unnecessary failures.
- **Theory: Ortega et al.**, "delusions in sequence models" (arXiv 2110.10819).
  - When the expert's behaviour depends on a hidden variable (here, the script's "this episode misses" draw), the sequence model
    treats its own sampled actions as evidence about it, so the miss becomes self-fulfilling.
  - This fits the imagined miss we observe.
- **No paper found** that reports an "unnecessary first-attempt miss rate" on clean starts.

## Synthesis for our data

1. **Perfect BC reproduces the marginal.** It should miss first in up to 23 % of episodes (59/253 demos). Our census: the final model
   (20000 steps) 12/72 (17 %, approaching the marginal), the same model at 12000 steps 2/72, the no-noise control 9/24
   (37 %). The no-noise rate is above the marginal, which
   suggests an extra mechanism, e.g. copying the history.
2. **Not supervising the failure segment is standard practice.** Variants:
   - relabel with the expert's intended action (DART/DAgger);
   - down-weight (ChauffeurNet 1/10, Sirius 0);
   - negative value (EIL, RLIF);
   - a condition token (RECAP, RePO-VLA);
   - cut the prefix from the context too (TSER).
3. **External perturbations are standard practice too.** The cause of the failure must be visible (IntervenGen). Use both ideas.
4. **WAM-specific pitfall** (our inference): masking only the action loss leaves the video-prediction targets. The
   model can still imagine the miss and act on it, so the video targets of predictions that start before the failure is visible
   must be masked too, or a flag that is never set at inference must mark them.
5. **History shortcut.** A kept failure prefix can become the trigger for the correction. Randomly truncate the history, so the
   recovery is predictable from the current frame (empty gripper).
6. **Proportion.** Too much recovery data teaches a propensity for perturbation (ChauffeurNet).
7. **Gains can be small.** RACER attributes only about 2 % to its recovery data.

## Recommendation for scripted simulator data

- **Failures.** No deliberate misses. Make failures exogenous and visible:
  - push the object a few cm just before the jaws close;
  - lower friction briefly while lifting (slip);
  - DART noise on the executed approach poses.
- **Labels.** The scripted expert's action computed from the current true state.
- **Old scripted-miss data, if reused.** Mask both the action and the video targets from the miss decision until the failure
  is visible.
- **Simulator DAgger.** Run the learned policy and label every state it visits with the scripted expert (cheap in simulation).
- **Evaluation.** Report the unnecessary-miss rate on clean starts (target near 0) and recovery success under injected
  perturbations, separately.
