# Matched PushT comparison of visual SAC and fixed-eight Diffusion Policy

**Across 80 distinct starting scenes, the saved visual SAC policy completed 0 of 80 pushes. The original fixed-eight Diffusion Policy achieved 61.25% success when each scene received equal weight.** Mean final coverage was 6.71% for SAC and 91.52% for fixed eight. The original 40-scene comparison is reported separately below.

## Setup

We ran the final checkpoint from the [100,000-movement SAC training run](2026-11-05-robotics-visual-sac-training.md) once per scene in deterministic inference mode. That checkpoint was trained online using standard Stable-Baselines3 SAC, one training seed, and Gym-PushT rewards. The comparison reuses both saved original fixed-eight prediction draws for scenes 170–209 from the [Diffusion Policy confirmation](2026-10-17-robotics-recovery-confirmation.md). Those scenes were outside the SAC training and development seed ranges. We used the original pretrained Diffusion Policy as the comparator and kept both policies frozen.

Both policies use two camera frames and their pusher positions, absolute x–y movement commands, and a 300-movement limit. Success requires true target coverage strictly above 95%. We matched the initial physical state and rendered image for both Diffusion Policy draws in every scene. Initial coverage matches within `1e-12`, the existing tolerance for the final digits of the geometry calculation. Peak coverage includes the starting scene. For paired differences, we average the two Diffusion Policy draws within each scene and then resample the 40 scenes.

## Original 40-scene results

| Policy | Completed pushes | Mean final coverage | Mean peak coverage | Mean movements |
|---|---:|---:|---:|---:|
| Visual SAC, 100,000 training movements | 0/40 (0%) | 6.37% | 7.18% | 300.00 |
| Original Diffusion Policy, fixed eight | 46/80 (57.5%) | 90.96% | 93.97% | 228.44 |

All SAC attempts reached the movement limit. SAC briefly improved on starting coverage in six scenes and finished above starting coverage in four. Its best peak was 37.65% on scene 171. The paired success difference (SAC minus fixed eight) was −57.5 percentage points, with a 40-scene bootstrap interval of −70.0 to −45.0 percentage points. This interval describes variation across starting scenes for the two saved policies. The result concerns one SAC training seed and the stated training budget.

## Expansion to 80 distinct scenes

The expansion adds 40 new deterministic SAC attempts on scenes 110–149. It uses the same SAC checkpoint, observations, movement limit and success rule. The matching original fixed-eight attempts are from the [staged-extension evaluation](2026-10-15-robotics-staged-extension.md), with one saved prediction draw per scene. All 40 new physical states and camera images matched their archived counterparts. The earlier 40 SAC attempts on scenes 170–209 were reused. Both scene sets were outside SAC training and development.

| Scenes | SAC successes | Fixed-eight successes | SAC mean final coverage | Fixed-eight mean final coverage |
|---|---:|---:|---:|---:|
| 110–149 | 0/40 (0%) | 26/40 (65.0%) | 7.05% | 92.08% |
| 170–209 | 0/40 (0%) | 46/80 (57.5%) | 6.37% | 90.96% |
| All 80, equal weight per scene | 0/80 (0%) | 61.25% | 6.71% | 91.52% |

Fixed eight completed 72 of 120 recorded attempts overall. The pooled percentage in the table first averages the two draws within each scene for 170–209, then averages across all 80 scenes. Mean peak coverage was 7.27% for SAC and 94.34% for fixed eight. The best peak among the 40 new SAC attempts was 36.91% on scene 145. The larger comparison gives broader scene coverage for the same single SAC training seed and budget.

## Reproduction

Use the existing `.venv-robotics`, the isolated SAC dependencies, and the local checkpoint listed in the [training record](../results/robotics-visual-sac-training/standard-01/movements-100000.json). Use a fresh output directory for a repeat.

```powershell
.\.venv-robotics\Scripts\python.exe -B experiments\robotics_visual_sac_comparison.py --output results\robotics-visual-sac-comparison-repeat
.\.venv-robotics\Scripts\python.exe -B experiments\robotics_visual_sac_expansion.py --output results\robotics-visual-sac-comparison-80-repeat
.\.venv-robotics\Scripts\python.exe -B -m unittest discover -s tests -p 'test_robotics_visual_sac*.py' -v
```

All six focused tests passed. A separate artifact check verified source and checkpoint hashes, all 40 starting-state matches, the 80 saved fixed-eight records, the movement and success rules, and the reported metrics and intervals. The [protocol](../results/robotics-visual-sac-comparison/protocol.json), [step traces](../results/robotics-visual-sac-comparison/sac-traces.json.gz), and [per-scene summary](../results/robotics-visual-sac-comparison/summary.json) record the comparison.

A separate check of the expansion verified the 40 new traces, their starting states, the reused results and the equal-scene averages. See the [expansion protocol](../results/robotics-visual-sac-comparison-80/protocol.json), [new SAC traces](../results/robotics-visual-sac-comparison-80/new-sac-traces.json.gz), and [combined summary](../results/robotics-visual-sac-comparison-80/summary.json).
