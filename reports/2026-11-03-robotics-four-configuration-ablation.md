# Four-configuration ablation: fine-tuned policy with fixed-eight execution

**The missing configuration is complete: fine-tuning with fixed-eight execution succeeded in 56/80 attempts (70%), while the same fine-tuned model with adaptive execution succeeded in 60/80 (75%).** Adaptive execution used 18.62% fewer predictions than fine-tuning alone. The additional four successes remain uncertain; the stronger evidence for adding the controller is prediction efficiency. Average final alignment changed little.

This completes the proposal's two-by-two comparison: original or fine-tuned weights, each with fixed-eight or adaptive 8/12 execution. It supports evaluating both components without assuming that their combination must win every metric.

## Scope and fixed conditions

Only the missing trained/fixed-eight condition was collected: **80 new scored attempts** on scene seeds 170–209, with prediction-draw offsets 60000 and 70000. The other three conditions reuse all 240 attempts from the [completed confirmation](2026-10-17-robotics-recovery-confirmation.md). These are forty initial scenes and two stochastic prediction samples per scene, not eighty independent scenes. The earlier twenty-scene training pilot is not pooled into this comparison.

The checkpoint is exactly the previous final-update-600 output block. No training, checkpoint selection, controller tuning or generation change occurred. All conditions use the pinned public architecture, H16 generation, two observation frames, 100 denoising steps and a 300-movement limit. Success requires true target coverage strictly greater than 95%. Failed attempts remain in every average.

The new condition uses the unchanged fixed-eight episode collector. It executes eight predicted actions and then predicts again; it does not activate the additional camera monitor, action extensions or early interruptions. The policy itself retains its original camera and pusher-position feedback. Separate condition bindings distinguish its trained weights from the original fixed-eight model even though both inner collectors are named `fixed8`.

Before scored inference, the [protocol](../results/robotics-recovery-fixed8/protocol.json) froze the schedule, checkpoint, source/test hashes, previous results and statistics. A new collection wrapper and CPU tests reuse existing inference and physics code; the original confirmation and its sources remain unchanged.

## Complete results

Each row contains eighty attempts on the same scene/sample pairs. Runtime averages span collection sessions and are descriptive rather than a controlled speed benchmark.

| Configuration | Successful pushes | Average final alignment | Average predictions | Average runtime |
|---|---:|---:|---:|---:|
| Original model, fixed eight | 46/80 (57.5%) | 90.96% | 29.0250 | 143.26 s |
| Original model, adaptive 8/12 | 48/80 (60.0%) | 91.59% | 25.3375 | 135.95 s |
| Fine-tuned model, fixed eight | 56/80 (70.0%) | 92.01% | 27.6625 | 136.07 s |
| Fine-tuned model, adaptive 8/12 | 60/80 (75.0%) | 91.88% | 22.5125 | 120.92 s |

Compared with the original fixed-eight baseline, fine-tuning alone achieved ten more successes, 4.69% fewer predictions and 5.02% lower recorded runtime. These are observed averages: its success interval reaches zero and its prediction/runtime intervals include zero.

Adding adaptive execution to the fine-tuned model achieved four more successes, **5.15 fewer predictions per attempt (18.62%)** and 15.15 seconds lower recorded runtime (11.13%). It saved 412 predictions over the eighty matched attempts. Mean final alignment was 0.13 percentage points lower, with substantial uncertainty in that difference.

The combined method retains the highest observed success count and lowest prediction/runtime averages. Fine-tuning alone has the highest point estimate of final alignment; the alignment differences do not establish superiority.

### Both prediction samples

| Draw offset | Configuration | Successes | Average predictions | Average runtime |
|---|---|---:|---:|---:|
| 60000 | Original, fixed eight | 22/40 | 29.400 | 145.14 s |
| 60000 | Original, adaptive | 24/40 | 25.575 | 137.12 s |
| 60000 | Fine-tuned, fixed eight | 28/40 | 27.975 | 137.51 s |
| 60000 | Fine-tuned, adaptive | 31/40 | 22.200 | 119.66 s |
| 70000 | Original, fixed eight | 24/40 | 28.650 | 141.37 s |
| 70000 | Original, adaptive | 24/40 | 25.100 | 134.79 s |
| 70000 | Fine-tuned, fixed eight | 28/40 | 27.350 | 134.63 s |
| 70000 | Fine-tuned, adaptive | 29/40 | 22.825 | 122.17 s |

The combined method has more successes and fewer predictions in both draws than fine-tuning alone. These draws reuse the same forty physical scenes and are not independent forty-scene replications.

## What the ablation establishes

The four configurations answer four distinct questions:

- Original adaptive versus original fixed-eight measures the controller's effect with the original model.
- Fine-tuned fixed-eight versus original fixed-eight measures the complete fine-tuning recipe under the original execution strategy.
- Fine-tuned adaptive versus original adaptive measures fine-tuning under the adaptive controller.
- Fine-tuned adaptive versus fine-tuned fixed-eight measures the controller's additional effect after fine-tuning.

For uncertainty, both draws are averaged within each scene before bootstrapping forty paired scene means. The protocol fixes 10,000 resamples and seed 20261103. These intervals are descriptive for this ablation completion; they do not account for between-session hardware effects or repeated training seeds. The bootstrap seed differs from the original confirmation, so its preserved intervals need not have identical Monte Carlo endpoints.

| Contrast | Metric | Observed difference | Paired scene-bootstrap 95% interval |
|---|---|---:|---:|
| Fine-tuned fixed-eight minus original fixed-eight | Success rate | +12.50 pp | 0.00 to +25.00 pp |
| Same | Final alignment | +1.04 pp | -1.40 to +3.40 pp |
| Same | Predictions | -1.3625 | -3.6625 to +0.9500 |
| Same | Recorded runtime | -7.19 s | -18.72 to +4.39 s |
| Fine-tuned adaptive minus fine-tuned fixed-eight | Success rate | +5.00 pp | -8.75 to +18.75 pp |
| Same | Final alignment | -0.13 pp | -3.60 to +3.25 pp |
| Same | Predictions | -5.1500 | -7.8375 to -2.4375 |
| Same | Recorded runtime | -15.15 s | -28.71 to -1.42 s |

Here, pp means percentage points. The controller's prediction-saving interval remains below zero after fine-tuning. Its additional success-rate interval includes zero, so four more successes do not establish a reliable success advantage. The recorded runtime interval also lies below zero, but collection-session differences prevent interpreting it as a controlled timing effect.

Fine-tuning alone gained eighteen matched successes and lost eight relative to the original fixed-eight baseline. The combined method gained seventeen and lost thirteen relative to fine-tuning alone. These are scene/draw pairs, not distinct scenes; the full lists are retained in the [summary](../results/robotics-recovery-fixed8/summary.json). We cannot claim that either change preserves every previously successful attempt.

**Interpretation:** the point estimates suggest that fine-tuning supplies the larger success improvement, while adaptive execution supplies clearer additional prediction savings. The data support retaining the combined configuration as the project's main evaluated recipe. They do not prove that either component is necessary in every scene, that the combined method universally dominates, or that recovery data alone caused the training benefit. No controller or checkpoint is automatically replaced.

## Validation, resumption and resources

All **711 CPU tests passed** before the protocol was frozen, including eighteen new checks for trained/fixed-eight weight identity, unchanged backbone, original-source preservation, schedule completeness, safe resumption and forty-scene statistics. The first suite invocation encountered fourteen Windows temporary-directory permission errors in older tests; rerunning with a workspace-local temporary directory passed without changing the frozen implementation.

Three short sixteen-action startup checks verified original, trained and restored-original execution. Original actions and physical states reproduced the saved controls. Every new scored attempt also matched the existing trained/adaptive attempt's first eight predicted commands and its executed prefix up to the first existing replanning point. Actions, physics coordinates and RGB hashes match exactly; the existing 1e-12 tolerance applies only to simulator coverage. Thus changing the execution rule did not accidentally change the checkpoint, initial scene or initial prediction sample.

Collection was interrupted after **55 saved attempts**. Strict CPU verification passed before resuming at attempt 56. All completed attempts were retained; no completed attempt was rerun, replaced or excluded. The [resumed invocation record](../results/robotics-recovery-fixed8/evaluation-resources-from-055.json) covers only attempts 56–80, model loading and verification: 60.26 minutes, 2.55 GiB sampled peak process RAM and 2.99 GiB minimum available system RAM. The interrupted first segment left no invocation resource record, so these are not whole-run RAM or elapsed-time measurements.

Across all eighty scored attempts, the new condition executed 17,439 movements and made 2,213 predictions. Summed attempt time was 181.42 minutes, excluding loading, startup checks, trace writing and the interruption gap. Maximum recorded per-attempt allocated GPU memory was 1,095.41 MiB. Episode-weighted mean fresh-prediction time was 4.8757 seconds, versus 4.8748 seconds for the trained/adaptive reference; the observed saving comes from fewer calls rather than a smaller neural computation budget.

The separate [raw-artifact audit](../results/robotics-recovery-fixed8/artifact-audit.json) checked all 320 model identities, 70,797 recorded movements, eighty new matched prefixes, every table entry and all five paired comparisons. It independently recalculated statistics from JSON and compressed traces without importing the collector, running inference or replaying physics. It also verified that prior evidence and frozen source hashes remain unchanged.

## Reproduction and limits

Use the project Python 3.11 environment and the exact trained artifact produced by the [training pilot](2026-10-16-robotics-recovery-pilot.md). The [original confirmation](2026-10-17-robotics-recovery-confirmation.md) is required for the reused reference arms. Existing completed results are preserved when running validation or summarization:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8.py --stage verify
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8.py --stage summarize
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8_audit.py
```

For a separate reproduction, use the dedicated ignored repeat directory. This repeats the missing eighty attempts on the same inspected scenes; it is not a new generalization cohort:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8.py --stage prepare --output-dir results\robotics-recovery-fixed8-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8.py --stage run --output-dir results\robotics-recovery-fixed8-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_fixed8_audit.py --output-dir results\robotics-recovery-fixed8-repeat
```

Do not run concurrent GPU collectors. No new model or method should be tuned on these scenes and then described as unseen evaluation. This comparison uses one trained checkpoint in clean, paused two-dimensional simulation. It does not establish recovery-data-specific causation, real-camera robustness, real-time deployment or physical-robot performance. The adaptive component includes the complete visual/path controller; this experiment does not separate extension from interruption. Preserve the designated milestone `0f40449`, the exact final600 checkpoint and the three-arm confirmation as existing evidence.
