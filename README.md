# V-JEPA kitchen

The continuation of [microwave-from-human-video](https://github.com/HanshangZhu/microwave-from-human-video):
the human video trains the world model; the digital twin only examines it.

## Why

In the first project, three RGB-D videos of me putting a pot into a microwave became a
scripted-solver dataset and an imitation policy in a measured digital twin. Its clearest
negative result: wherever a scripted oracle exists, human motion used as *action labels*
loses to it (59 % vs 88 % at matched demonstration counts). Its clearest positive one: the
video carries what the simulator cannot produce -- how a real door, pot and room look and
move under manipulation -- and that needs no action labels to learn.

So here the human data changes sides of the pipeline:

1. **Scale ego-motion video**: head- or chest-mounted RGB(-D), many takes, many objects and
   rooms; the wrist pose tracked as the action proxy.
2. **Train the video head of a world-action model on it** (V-JEPA 2-style predictor, or the
   video branch of a WAM); where data are few, the object-centric variant -- predict pot pose
   and door angle from wrist motion -- is the minimal instance.
3. **Align the action head with the robot's own embodiment**: SFT on robot demonstrations,
   then RL in the twin, where success is measurable and the oracle is a critic, not a
   demonstrator.
4. **The simulator is the exam**: the twin and its Gaussian-splat room score the model on
   starts, rooms and bodies the video never showed.

## Plan

| step | what | status |
|---|---|---|
| 0 | object-centric world model on the three existing takes (state: pot pose + door angle; action: wrist pose); evaluated in the twin against simulated object states | next |
| 1 | ego-video capture at scale (head-mounted RealSense / phone), wrist tracking | planned |
| 2 | V-JEPA 2 / WAM video head fine-tuned on the captures; prediction quality vs horizon | planned |
| 2a | LingBot-VA (a world-action model, Wan2.2 5B) LoRA post-trained on twin demonstrations rendered over the room splat, with the 49 cleaned ego takes as video-only clips; closed loop in the twin | first working model: 71/72 held-out starts ([Results](#results-step-2a)); the ego video gave no gain so far |
| 3 | action head: SFT on simulator demonstrations, RL in the twin | planned |
| 4 | whole-body G1 in Isaac Lab on an existing whole-body controller | planned |

## Results (step 2a)

**Task.** In the twin, a Franka arm opens the microwave, puts the pot inside and closes the door;
success = the pot is inside and the door is shut. The pot's start position is drawn from the range
of the demonstrations. The model sees only the front camera (160 x 288, from a held-out take's
registered pose) with the room splat behind the scene, and is queried every 32 simulator steps
(4 latent frames); its predicted end-effector pose is executed by IK; episodes stop at 1500 steps.
Starts 0..7 picked checkpoints; starts 8..79 never did.

**Model.** `lingbot-va-base` with LoRA rank 128, the pre-training action layout (end-effector pose,
joints, gripper), attention over the whole episode in half of the samples. Data: 260 solver
demonstrations (7 held out for validation) -- whole task (59, plus 20 with DART noise), starts in
the middle of the task (76), recovery from a failed place (46) and from a missed door handle (59). **Noise on the action
history**: in half of the training samples the past actions the model attends to are heavily
noised, as LingBot-VA's recipe already does for the past video (`train_va.py
--action-noisy-cond-p 0.5`). 14000 steps at lr 1e-4, then continued to 20000 with the lr falling
linearly to 1e-5. 59 h on one RTX 5090.

| model (one machine, the same starts for every row) | starts 8..31 | starts 32..79 | all 72 held-out starts |
|---|---|---|---|
| **the model above, 20000 steps** | 24/24 | 47/48 | **71/72 (99 %, 95 % CI 93-100)** |
| the same at 12000 steps (constant lr; picked on starts 0..7) | 16/24 | 39/48 | 55/72 (76 %, 65-85) |
| the same data without the action-history noise (best of 7 checkpoints) | 5/24 | | |
| with the noise, plus the ego takes as 30 % of samples (best of 7) | 11/24 | | |
| earlier best: slower demonstrations, a wrist camera, pot-grasp retry demos, no noise, 20000 steps (111 h), queried every 8 steps | 14/24 | 29/48 | 43/72 (60 %, 48-70) |

Before these: zero-shot 0/21; the first post-trained models 0-1 of 8.

What the comparisons show (paired by start, exact McNemar):

- **The action-history noise is what made the policy work**: 16/24 with it, 5/24 without, everything
  else identical (13 starts solved only with it, 2 only without; p = 0.007). Without it 18 of the 19
  failures are missed pot grasps while the door, which never moves, is still opened. An earlier
  diagnostic (`reliance_va.py`) found the predicted motion 2-2.8x more sensitive to the action
  history than to the video in an early model trained without the noise: it continues its own
  trajectory instead of looking where the pot is.
- **Longer training with a falling lr** took the same recipe from 76 % to 99 % (17 starts gained,
  1 lost; p = 0.0001). Steps and lr schedule changed together.
- **The ego video did not help here**: 11/24 with it against 16/24 without (p = 0.27), one run with
  43 takes as video-only samples. The README's premise is not supported yet.
- **Rates from 24 starts do not rank models.** The same weights on the same 24 starts gave 19/24 on
  one machine and 14/24 on another (the machines differ numerically in the last digits, and
  closed-loop rollouts amplify that). Results here are therefore over 72 starts and compared on one machine.

Not shown yet: the final model on the second machine, a second training seed, steps vs lr decay
separately, anything outside this task, scene and camera.

## Contents

- `wam/` -- the LingBot-VA work (step 2a): `render_va.py` renders solver demonstrations from a
  take's camera over the room splat, `encode_va.py` encodes robot runs and ego takes to VAE latents,
  `train_va.py` is the LoRA post-training (`--action-noisy-cond-p`, `--lr-end`, `--chunk-max`),
  `eval_va.py` the closed loop in the twin (`--chunk`: latent frames per query), `average_lora.py`
  averages a run's checkpoints; `imagine_va.py`, `prefix_va.py`, `action_acc_va.py`,
  `reliance_va.py` and `head_consistency_va.py` are diagnostics (imagined video from a start or a
  demonstration prefix; action error with the true history; how much the actions follow the action
  history vs the video; agreement of the pose and joint outputs).
- `sim/`, `assets/`, `recon/gs_server.py`, `recon/gs_kitchen.py` -- the twin, the scripted solver
  and demonstration generator, and the splat renderer, copied from the first repository on
  18 October 2026 and developed here since (the solver's `--smooth`, `--miss-back`, `--speed`).

Registration and per-frame object extraction stay in the first repository. The room splat and
the takes are recordings of my home and are shared on request (zcabhz4@ucl.ac.uk); scripts expect
them, and all generated data, under `Gitignore-but-important/` (not in git).

*Started 17 October 2026.*
