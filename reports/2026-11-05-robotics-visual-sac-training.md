# Visual SAC on PushT after 100,000 training movements

**Visual SAC completed 0 of 8 development pushes after 100,000 training movements.** Mean final coverage fell from 12.58% at initialization to 4.73%. The laptop completed the planned run, but this configuration and budget did not produce a usable policy for the formal comparison.

## Setup

The run started from fresh weights with the same seed and Stable-Baselines3 2.7.1 SAC settings as the [ten-minute pilot](2026-11-05-robotics-visual-sac-pilot.md). It trained online using Gym-PushT's native coverage reward and a replay buffer of 20,000 transitions with batches of 64. Each policy observation contained two 96 × 96 RGB frames and their corresponding pusher positions. Each action gave one absolute x–y pusher target, using the same movement command as Diffusion Policy. Training used zero expert demonstrations. The policy received no simulator block pose or coverage as input.

The run was capped at 100,000 movements or 75 minutes. It scored the deterministic actor on the same eight development scenes (0–7) before training and after 25,000 and 50,000 movements. The continuation rule required either a success or higher mean final coverage than initialization at either intermediate checkpoint. The 25,000-movement score cleared that rule. Each evaluation allowed up to 300 movements and counted success when true coverage exceeded 95%. Training reset seeds came from `[100000, 1000000)`.

## Results

| Training movements | Successes | Mean final coverage | Mean peak coverage |
|---:|---:|---:|---:|
| 0 | 0/8 | 12.58% | 13.93% |
| 25,000 | 0/8 | 14.07% | 14.12% |
| 50,000 | 0/8 | 4.67% | 4.74% |
| 100,000 | 0/8 | 4.73% | 4.74% |

Mean return over recent online training episodes rose from 0.66 at 25,000 movements to 26.65 at 100,000. Those returns come from the exploratory policy on training scenes. The deterministic actor completed zero pushes at every development checkpoint. The evidence covers one training seed and eight previously inspected scenes. The formal comparison was deferred.

The run reached its movement limit after 99,000 updates and 66 minutes 34 seconds from the end of the initial evaluation, including checkpoint saves and scoring. Peak sampled process RAM was 3.21 GiB, minimum sampled available system RAM was 2.08 GiB, and peak PyTorch GPU allocation on the GTX 1650 was 124.44 MiB. A retry was briefly started when the original process was absent from the process listing. The original run was still active, so the retry was stopped and its partial records were archived locally. Treat the elapsed time as descriptive because of the overlap.

## Reproduction

Use the pinned dependencies and `.venv-robotics` described in the [pilot report](2026-11-05-robotics-visual-sac-pilot.md). Choose a fresh run ID to preserve existing results.

```powershell
.\.venv-robotics\Scripts\python.exe -B experiments\robotics_visual_sac_training.py --run-id standard-repeat
.\.venv-robotics\Scripts\python.exe -B -m unittest discover -s tests -p 'test_robotics_visual_sac*.py' -v
```

All six focused tests passed. A separate check recomputed aggregate scores, matched all eight initial scenes to the earlier fixed-eight records, and verified the stopping rule and source and checkpoint hashes. The final checkpoint reloaded on CPU and returned a valid target. See the [protocol](../results/robotics-visual-sac-training/standard-01/protocol.json) and [per-scene results and summary](../results/robotics-visual-sac-training/standard-01/summary.json). Each checkpoint is about 36.29 MiB and is saved locally without replay in `tmp/visual-sac-training-models/`.
