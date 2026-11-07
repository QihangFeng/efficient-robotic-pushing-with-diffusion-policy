# Lost baseline successes: diagnosis and rehearsal pilot

**Neither preservation recipe reduced negative flips on this seven-scene development comparison.** Original fixed-eight and the current recovery-trained/adaptive method each succeed in **11/14 attempts**. Baseline-success rehearsal succeeds in **8/14**; the subsequent output-preservation candidate succeeds in **10/14**. The latter repairs three current-method failures but loses four current successes, increasing original-baseline losses from two to four. Neither candidate is adopted; the established checkpoint, controller and forty-scene results remain unchanged.

The preceding saved-trajectory diagnosis does not support treating all nine original losses as catastrophic forgetting or as one faulty interruption rule. Seven of the nine attempts succeed with the same fine-tuned model under fixed-eight execution. Their combined-method losses include destructive fresh predictions and final positioning stalls. The mitigation remains unresolved.

The combined method still gains 23 successes and loses nine against original fixed eight, giving 60/80 versus 46/80. Those nine scene/prediction-draw pairs cover **seven distinct starting scenes**. Preserving the gains matters as much as recovering selected losses.

## Scope and checks

Sections 1–3 analyze the existing four-configuration comparison on scenes 170–209, draw offsets 60000/70000. That CPU-only diagnosis reads all **320 attempts and 70,797 recorded movements**, rather than inspecting failures alone. Its review and auditor import no policy, training code or simulator and perform no new inference, training or physics. Sections 4 and 5 separately record the approved preservation pilots, which train separate output blocks and collect development attempts.

The [review](../results/robotics-baseline-loss-review/summary.json) binds 724 source/evidence files and verifies sidecars, compressed trace hashes, model identities, H16 generation, prediction accounting, first-success termination and common starting states. Existing frozen audits supply the earlier controller/queue provenance; this review does not claim to repeat neural inference or physics. The [separate arithmetic audit](../results/robotics-baseline-loss-review/artifact-audit.json) checks the component outcomes, peaks, stationary tails, matched-prefix indices and all comparison-group decline counts directly from raw JSON and traces.

Dynamic simulator geometry and true coverage are **offline diagnostics only**. Recorded camera estimates are identified separately. The unchanged runtime controller still uses its existing visual/path inputs. The designated milestone remains `0f40449`.

## 1. Separating weight changes from execution changes

Original fixed eight succeeds and fine-tuned/adaptive fails in every row below. The other two configurations separate different observed outcome patterns; they do not, by themselves, establish a causal mechanism.

| Original/adaptive succeeds? | Fine-tuned/fixed-eight succeeds? | Lost attempts | Interpretation to investigate |
|---|---|---:|---|
| No | Yes | 5 | The adaptive execution setting is associated with failure under either checkpoint. |
| Yes | Yes | 2 | Each component separately succeeds; their combination fails. |
| Yes | No | 1 | The fine-tuned checkpoint fails under either execution setting. |
| No | No | 1 | Both separate changes fail; original fixed eight alone succeeds. |

Thus **seven of nine** losses have a successful fine-tuned/fixed-eight control, while two also fail with fine-tuning alone. Calling all nine losses forgetting would overlook the execution interaction. Returning to fixed eight everywhere would also discard gains: over all eighty attempts, fine-tuned/fixed-eight succeeds 56 times versus 60 for the combined method and uses more predictions.

### Whole-cohort retention matters

Recounting all four configurations from the [confirmation index](../results/robotics-recovery-confirmation/evaluation.json) and [trained/fixed-eight index](../results/robotics-recovery-fixed8/evaluation.json), after verifying their hashes against this review's frozen bindings, gives:

| Configuration | Successes / 80 | Gains versus original fixed eight | Losses versus original fixed eight |
|---|---:|---:|---:|
| Original / fixed eight | 46 | Reference | Reference |
| Original / adaptive | 48 | 19 | 17 |
| Fine-tuned / fixed eight | 56 | 18 | 8 |
| Fine-tuned / adaptive | 60 | 23 | 9 |

Gains come from the original baseline's 34 failures; losses come from its 46 successes. Fine-tuned/fixed-eight rescues seven of the combined method's nine selected losses, but has **eight losses overall**, including six different original successes retained by the combined method. Switching execution settings exchanges failure cases; it does not reduce the whole-cohort loss count to two.

The original checkpoint itself loses **17** fixed-eight successes when paired with adaptive execution. Therefore, even exact preservation of that checkpoint's denoising outputs would not guarantee preservation of the fixed-eight baseline's closed-loop successes. The current fine-tuned/adaptive model already reduces that count to nine while gaining more successes. This limits the original-teacher preservation hypothesis; the new bounded test must be judged against the complete current method, not just closeness to original weights or a selected repaired case.

## 2. What happens in each saved combined attempt?

All nine combined attempts time out at 300 movements without ever crossing 95% true coverage. Values below are rounded for display; success uses the unrounded, strictly-greater-than-95% criterion.

| Scene / draw | Original/adaptive | Fine-tuned/fixed eight | Combined peak coverage / movement | Combined final coverage | Interruptions |
|---|---|---|---:|---:|---:|
| 177 / 70000 | Success | Fail | 91.56% / 140 | 32.90% | 1 |
| 179 / 60000 | Fail | Success | 94.23% / 192 | 81.78% | 1 |
| 184 / 70000 | Fail | Success | 85.62% / 188 | 83.45% | 0 |
| 190 / 60000 | Fail | Fail | 93.73% / 287 | 93.73% | 1 |
| 190 / 70000 | Fail | Success | 90.53% / 256 | 87.29% | 2 |
| 198 / 70000 | Success | Success | 94.97% / 157 | 94.58% | 0 |
| 202 / 60000 | Success | Success | 92.53% / 143 | 83.18% | 0 |
| 202 / 70000 | Fail | Success | 92.80% / 155 | 89.30% | 2 |
| 204 / 60000 | Fail | Success | 89.65% / 233 | 89.65% | 0 |

![Saved coverage histories for all nine lost attempts](../results/robotics-baseline-loss-review/lost-success-coverage.png)

Curves end when the recorded attempt ends. The red dashed line marks 95%; the plot shows simulator coverage, not a runtime controller signal. Different trajectories after divergence are not matched-state alternatives.

### The first timing difference is not a failure explanation

For each lost pair, the two fine-tuned configurations execute exactly matching actions and physical/image states through **movements 1–8**. After movement 8, adaptive execution retains the existing plan; fixed eight predicts again before movement **9**. The adaptive run executes cached movements **9–12**. Its true coverage is unchanged during those four movements in all nine cases.

Movement 9 is a common timing difference: it occurs in **74 of all 80 pairs**, including successes. Five pairs first differ at movement 17 and one at movement 33. A shared early divergence is therefore not a usable failure label. Unchanged early coverage also does not prove the suffix harmless: object orientation, approach direction and later visited states can matter.

### Fresh predictions can worsen alignment

- **177 / 70000:** an interruption occurs after movement 144. The fresh prediction beginning before **145** lowers true coverage from **79.72% after 144 to 28.71% after 152**. This establishes a destructive recorded prefix, not that continuing the discarded plan or substituting the original policy would have succeeded.
- **179 / 60000:** after ordinary trimming at 208, the fresh prediction executes **209–216** and lowers coverage from **92.31% to 76.45%**. Interruption after 216 does not ensure recovery.
- **202 / 70000:** the fresh prediction beginning before **185** lowers coverage from **87.55% after 184 to 60.39% after 190**, where it is interrupted.

Simply asking the same model for another prediction is not an established recovery mechanism.

### Several attempts stall despite repeated predictions

| Scene / draw | Stationary block after movement | Remaining movements | Further predictions | Final coverage |
|---|---:|---:|---:|---:|
| 184 / 70000 | 214 | 86 | 10 | 83.45% |
| 198 / 70000 | 221 | 79 | 9 | 94.58% |
| 204 / 60000 | 233 | 67 | 8 | 89.65% |

Stationarity uses the earlier review's `1e-9` tolerance for block-origin translation and wrapped rotation per movement. The pusher continues moving. In the scene-198 example it travels about **187.98 simulator coordinate units** during the stationary tail. These observations concern block motion, not measured contact forces.

The near-threshold scene 198 also illustrates measurement limits: after movement **180**, camera-estimated coverage is **95.16%**, while simulator coverage is **94.97%**. Camera-estimated completion cannot replace the fixed scoring criterion.

## 3. Successful controls rule out simple decline labels

The three-percentage-point screen below is inherited from the earlier retrospective decision review, not selected to optimize these outcomes. A fresh-prefix window contains up to the first eight executed movements of one prediction, ending sooner if interrupted or terminated. An extension window contains up to its four cached extra movements.

| Combined-method outcome versus original fixed eight | Attempts | Extensions with >3 pp drawdown / all extensions | Attempts with a fresh-prefix >3 pp drawdown |
|---|---:|---:|---:|
| Both succeed | 37 | 6 / 253 | 28 / 37 |
| New success | 23 | 1 / 181 | 18 / 23 |
| Lost success | 9 | 0 / 98 | 9 / 9 |
| Both fail | 11 | 2 / 107 | 11 / 11 |

None of the 98 extensions in the nine losses causes an immediate >3 pp overlap drawdown. The largest recorded extension-window drawdown in these losses is **2.52 pp**. Earlier extensions may still affect later states; these data do not establish their causal benefit or harm.

All nine losses contain a >3 pp fresh-prefix drawdown, but so do **46 of the 60 successes**. Four losses have no interruption at all. These findings do not justify reopening the rejected warning/weighted filters, treating every decline as failure or adding an oracle success/failure gate.

## 4. Baseline-success rehearsal pilot

The literature distinguishes improving average performance from preserving previously correct behavior. [Positive-Congruent Training (CVPR 2021)](https://arxiv.org/abs/2011.09161) gives greater preservation weight to examples handled correctly by the old classifier. Its classification method is not a ready-made trajectory controller. [Learning without Forgetting](https://arxiv.org/abs/1606.09282) preserves old model responses during adaptation. [FDPP (ICRA 2025)](https://arxiv.org/html/2501.08259v1) tests diffusion-policy fine-tuning on PushT and uses a distribution-preservation penalty; its preference-based RL pipeline differs from our supervised update.

**The approved pilot tests baseline-success rehearsal:** include native, successful original-policy trajectories alongside the existing expert/recovery data. This adapts the preservation idea; it is not a reproduction of PCT/FDPP, and training completion alone does not validate the mitigation.

We already have **20 ordinary successful baseline trajectories on discovery scenes 0–29**, prepared as diagnostic controls. They supply **2,686 complete native H16 windows** and were **not used to train the established recovery-trained checkpoint**. The separate supplement supplies the new candidate's training/validation split while leaving the diagnostic manifest intact. Its audit verifies the existing arrays and source metadata.

The fixed candidate recipe is:

1. Start from the same original checkpoint and update only the same final convolution block for 600 updates, retaining the learning rate, H16 labels, normalization and 100-step inference.
2. Keep four accumulated samples per update. Replace one of the two expert samples with a successful-baseline sample: **one expert + one baseline-success + two recovery**, versus the original **two expert + two recovery**. This preserves the recovery sample count and total denoiser training-call budget.
3. Split the twenty baseline-success scenes into sixteen training scenes and four validation scenes before training. Preserve the existing whole-scene recovery split. Exclude all forty evaluation scenes, their trajectories and the nine inspected losses from training/selection data.
4. Leave the milestone controller unchanged. There is no additional model or prediction call during deployment. A retention dataset may improve predicted movements; it is not guaranteed to resolve the seven execution-associated losses.
5. Compare the candidate, current combined model and original fixed-eight policy on a fixed development screen. Four held-out baseline-success scenes plus the three existing recovery-validation scenes, with two new prediction draws, give **42 development attempts**. The pre-run estimate was roughly **90 minutes plus loading/checks**. The exact split, draws, checkpoint and stopping rule were frozen before running it.

The development screen reports gains and losses relative to original fixed eight, overall success, coverage, predictions and runtime, including failures. A candidate should advance only if it improves the retention trade-off without sacrificing the existing recovery behavior. Confirming generalization would then require a separately agreed, untouched evaluation cohort. The inspected scenes 170–209 cannot become fresh evaluation evidence for this redesign.

The isolated [retention runner](../experiments/robotics_retention_pilot.py) implements this prescribed candidate. All **16 CPU runner tests pass**, including exact resumption on a tiny test model, candidate checkpoint identity, protected outputs and scene-cluster statistics. Its [execution protocol](../results/robotics-retention-pilot/protocol.json) freezes all 600 training updates and the 42 development attempts. The original two recovery samples and first expert sample match the previous training schedule at every update; only the second expert slot changes, using a separate baseline-sampling RNG. All prescribed training and comparison attempts are complete.

The explicit runner stages are `prepare`, `verify`, `train` and `evaluate`. Preparation and verification do not load the pretrained robot policy. Real training and pushing occur only in their named stages. More elaborate teacher-output anchoring remains a separate future option if rehearsal is inadequate; it is not part of this pilot.

This tests one prescribed data mix and one training seed. Replacing an expert sample halves the expert rehearsal count while keeping recovery examples and total updates fixed. It does not isolate the effect of adding baseline-success examples while preserving all expert rehearsal. The final update-600 checkpoint is fixed in advance; validation losses and development outcomes do not select another checkpoint or mixture.

### Training stage completed

The retention candidate completed **600 updates / 2,400 denoiser training forwards** in **207.51 seconds**, excluding policy loading. Peak allocated GPU memory was **1,111.08 MiB**. The new checkpoint is stored separately from the original and current recovery-trained heads. Its [training record](../results/robotics-retention-pilot/training.json) and [independent training audit](../results/robotics-retention-pilot/training-audit.json) verify 600 expert, 1,200 recovery and 600 baseline-success samples; all 1,800 unchanged sample slots reproduce the earlier recipe's choices. Only the same **1,313,282 final-block parameters** change, and the frozen backbone/normalization digest matches the original.

These checks establish a correctly executed training stage. The trained head's SHA-256 is `8a5cdddd67c7958a59513165dcdfd6e8b1a80380e95c383c89b5faeee6115bfc`.

On the fixed 32 validation windows per group, baseline-success denoising loss rose from **0.005654 to 0.008167**; expert loss rose from 0.002244 to 0.004628, while recovery loss fell from 0.018247 to 0.017308. These are noise-prediction losses on correlated windows, not pushing success rates or a checkpoint-selection rule. Baseline-success rehearsal did not preserve even this held-out denoising score under the chosen mix.

### Completed development comparison

All **42 scheduled attempts** finished in one collector invocation, with no completed attempt replaced, excluded or rerun. Each method used the same seven starting scenes and two prediction draws: **14 attempts per method, seven scene clusters**. All use two observations, H16 generation, 100 denoising steps, a 300-movement limit and success strictly above 95% simulator target coverage. Dynamic simulator state remains scoring-only. Both adaptive arms reuse the unchanged milestone controller.

The [summary](../results/robotics-retention-pilot/summary.json) includes all successes and failures:

| Method | Successful pushes | Average final alignment | Average predictions | Average runtime |
|---|---:|---:|---:|---:|
| Original pretrained / fixed eight | 11/14 (78.6%) | 91.60% | 25.93 | 127.30 s |
| Current recovery-trained / adaptive | 11/14 (78.6%) | 93.21% | 24.00 | 128.35 s |
| New baseline-rehearsal / adaptive | 8/14 (57.1%) | 88.79% | 26.86 | 144.35 s |

The candidate succeeds **4/7 in each draw**. Current and original each succeed **5/7 at offset 91000 and 6/7 at offset 92000**. These draws repeat the same seven physical scenes; they are not fourteen independent scenes.

| Method | Baseline holdouts: scenes 1, 5, 10, 23 | Recovery validation: scenes 15, 16, 18 | Gained / lost successes versus original fixed eight |
|---|---:|---:|---:|
| Original / fixed eight | 6/8 | 5/6 | Reference |
| Current / adaptive | 8/8 | 3/6 | 2 gained / 2 lost |
| New rehearsal / adaptive | 5/8 | 3/6 | 2 gained / 5 lost |

The new candidate retains the same two gains against original fixed eight, at **10/92000 and 23/91000**, but introduces three additional losses relative to the current method. All three time out at movement 300:

| Scene / draw | Current success at movement | Candidate final alignment |
|---|---:|---:|
| 1 / 92000 | 203 | 85.52% |
| 5 / 91000 | 294 | 93.53% |
| 23 / 92000 | 165 | 71.99% |

None of the current method's failed attempts is rescued. The candidate also retains the current losses against original fixed eight at **15/91000 and 16/92000**. Thus this recipe worsens retention on the baseline holdouts without an observed recovery-success improvement.

Uncertainty uses the frozen 10,000-resample, seven-scene bootstrap, after averaging both paired draw differences within each scene:

| Candidate minus current method | Observed difference | Paired-scene bootstrap 95% interval |
|---|---:|---:|
| Success rate | -21.43 percentage points | -42.86 to -7.14 pp |
| Final alignment | -4.42 percentage points | -9.16 to -0.29 pp |
| Predictions per attempt | +2.86 | +0.50 to +5.36 |
| Runtime per attempt | +16.01 s | +1.98 to +30.98 s |

These are descriptive development intervals for one prescribed mixture/training seed on seven inspected scenes. They do not establish population-wide inferiority of rehearsal, a unique cause of the losses or generalization to new scenes. Replacing an expert slot and changing the data source are coupled in this test. The original nine confirmation losses were not rerun or used to select this candidate.

**Decision:** preserve this candidate and its negative result as evidence; keep the established checkpoint and controller. The result does not justify tuning on the forty inspected evaluation scenes, adding inference candidates or reopening the rejected interruption filters. A further preservation method would require a separately discussed training recipe and bounded protocol.

### Cost and completed audit

The collector's [resource record](../results/robotics-retention-pilot/evaluation-resources-from-000-1794077221799475400.json) covers **94.54 minutes**, including the four startup checks and collection/verification overhead. The timer starts after initial policy/head loading and initial backbone hashing. Its generated scope label says "including load"; that phrase overstates the measured interval, and the original record is preserved. Per-attempt timing includes prediction, physics and camera/controller monitoring, including the initial camera estimate. Model loading, environment reset and artifact writing are outside those per-attempt measurements.

Peak allocated inference GPU memory is **1,095.41 MiB in every arm**. Episode-weighted average prediction times are 4.8746/4.8881/4.8991 seconds for original/current/candidate. The collector samples 1.12 GiB peak process RSS and at least 4.08 GiB available system memory during its measured interval. The candidate's unchanged inference architecture makes more predictions and takes longer per attempt in this screen.

The [independent complete audit](../results/robotics-retention-pilot/artifact-audit.json) passes: **42 scored attempts, four startup attempts and 9,304 recorded movements**. It rechecks frozen inputs/training provenance, model/head identities, exact original-head restoration, common physical/RGB initial states, action-queue indices, H16 accounting, success/timeout arithmetic, all means and gained/lost lists, and seven-scene bootstrap statistics. It reads saved artifacts and the small head tensors on CPU; it does not repeat gradients, neural inference or physics. The collector also verifies the frozen backbone and original weight file at completion.

Recheck the completed pilot without new training or pushing:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_retention_pilot.py --stage verify
.\.venv-robotics\Scripts\python.exe experiments\robotics_retention_pilot_audit.py --stage complete
```

### Prepared data and fixed schedule

The [separate data supplement](../results/robotics-retention-preflight/baseline-supplement.json) references the existing arrays without changing their original diagnostic metadata. All **3,006 RGB frames**, own pusher positions and **2,986 executed commands** match the twenty saved successful source traces. The supplement contains 2,686 complete H16 windows with two observations per sample and no padding. No image was re-rendered or re-fitted.

The whole-scene baseline split is selected by SHA-256 order with fixed key `20261107/baseline-retention`, before candidate outcomes:

- Baseline training scenes: **0, 8, 12, 13, 14, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29**.
- Baseline validation scenes: **1, 5, 10, 23**.
- Development scenes: **1, 5, 10, 15, 16, 18, 23**, combining baseline holdouts with existing recovery validation scenes.
- Prediction offsets: **91000/92000**. The three methods rotate within each scene/draw pair, giving the [42-attempt schedule](../results/robotics-retention-preflight/plan.json).

These are inspected development scenes, not a fresh generalization cohort. The prepared supplement excludes scenes 170–209. The original expert/recovery dataset, its splits, model checkpoints and previous evaluations remain unchanged. The preflight files record data preparation; the separate training and evaluation records establish what actually ran.

All ten preflight CPU tests and the [complete array/trace check](../results/robotics-retention-preflight/data-audit.json) pass. Repeating preparation reproduces identical JSON. The preflight has no training/evaluation command and imports no policy, optimizer or simulator. Run it with:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_retention_preflight.py
.\.venv-robotics\Scripts\python.exe -m unittest discover -s tests -p test_robotics_retention_preflight.py
```

## 5. Output-preservation pilot

**Qihang approved this second pilot after reviewing the unsuccessful rehearsal result.** Keep all of the established expert/recovery examples, and penalize the student when its denoising output changes too much from the original model on successful-baseline states. The aim is to retain old behavior while learning recoveries. The unsuccessful mixture replaced half the expert examples; this candidate preserves that expert rehearsal and adds an explicit preservation objective. Mitigation development now belongs on `experiment/negative-flip-mitigation`; the earlier six commits remain intact there, while `main` returns to the preceding `4ff40a4` project snapshot.

[Learning without Forgetting](https://arxiv.org/abs/1606.09282) motivates preserving old model responses during adaptation. [FDPP, section IV-C](https://arxiv.org/html/2501.08259v1) regularizes a diffusion policy toward its pretrained action distribution. The proposed denoiser-output penalty is our supervised adaptation of this principle. It is neither an exact KL calculation nor a reproduction of either paper, and it cannot guarantee identical closed-loop outcomes.

The reviewable bounded recipe is:

1. **Keep the original successful training recipe.** Start from the original pretrained checkpoint; use the same final block, 600 updates, learning rate `1e-5` and two expert + two recovery samples per update. Preserve all four original sample slots. Add one baseline-training window per update from the existing sixteen-scene split; isolate its random-noise generator so the four supervised slots retain their original noise stream.
2. **Add one fixed preservation loss.** Feed the same noisy H16 actions, two-frame observations and diffusion timestep to the current student and a frozen original teacher. Minimize the mean squared difference between their predicted noise tensors, with fixed weight **1.0** added to the existing mean supervised loss. This coefficient is a predeclared engineering choice, not a value validated by the cited papers. Do not use the baseline example's ordinary denoising-label loss as another supervised term. Do not search weights or select an intermediate checkpoint from development outcomes.
3. **Reuse the frozen backbone.** The installed LeRobot `DiffusionModel.compute_loss` calls the denoiser once, and `DiffusionConditionalUnet1d.forward` applies `final_conv` once before rearranging its output. Since the earlier backbone is frozen, the planned implementation can capture the final-block input and pass it through a separate frozen copy of the original final block. This avoids loading a second full model or generating teacher action plans. The budget is **3,000 student denoiser forwards plus 600 teacher final-block forwards**, followed by final-update-600 selection. Only the existing 1,313,282 student parameters change. The teacher is training-only; deployment retains the existing model and controller structure.
4. **Freeze and check before collection.** Use separate new source, protocol, checkpoint and output paths. First check penalty/gradient arithmetic, teacher immutability, preserved sample/noise streams and exact resumption on CPU. A single real-update feasibility check must restore the original student state and verify the frozen backbone before the prescribed training run. Stop on resource, nonfinite-gradient or provenance failures. Keep all old frozen sources and evidence intact.
5. **Collect fourteen candidate attempts.** Reuse the audited original/current 28 reference attempts from section 4. Run the candidate on the same seven development scenes and offsets 91000/92000, with the unchanged H16/100-step/300-movement/strict-95% protocol and four short original/current/candidate/original startup checks. Verify identical starting physical/RGB states and original restoration. Complete all fourteen candidate attempts regardless of interim outcomes. Reused references make cross-session runtime descriptive; success, gain/loss and prediction counts remain the main comparison.
6. **Judge retention and recovery together.** Report all newly gained/lost successes against both references, including any new current-method losses. Consider a separate fresh confirmation only if original-baseline losses fall below the current **two**, total success is at least **11/14**, and recovery-validation success is at least **3/6**. Bootstrap seven scenes after averaging their two draws. These inspected scenes remain development data; exclude scenes 170–209 from training and selection. No automatic promotion or further mixture/weight search follows this test.

The measured candidate collector averaged 144.35 seconds per attempt. Fourteen new attempts at that rate take about **34 minutes**; allowing for training and checks gives an estimated **35–50 minutes of model execution**, plus implementation time. The new training cost and GPU peak have not been measured. Reusing the backbone and keeping a small teacher head makes this a plausible laptop experiment, subject to the feasibility check. This remains a proposed mitigation, not a finding that negative flips have been solved.

### Implementation and frozen preparation

The separate [preservation runner](../experiments/robotics_preservation_pilot.py), [CPU tests](../tests/test_robotics_preservation_pilot.py) and [artifact auditor](../experiments/robotics_preservation_pilot_audit.py) preserve all previous experiment sources and results. All **20 CPU tests pass**. They check the zero penalty for an unchanged original head, the gradient pulling a perturbed output back toward its teacher, frozen teacher/backbone state, all 2,400 original sample slots, independent auxiliary noise, exact interrupted/resumed updates and RNG, native inputs, protected artifacts, reference trace arithmetic, candidate head selection and seven-scene statistics. The zero-weight CPU control also reproduces later original supervised updates exactly.

The [new protocol](../results/robotics-preservation-pilot/protocol.json) binds the source/tests/auditor, installed LeRobot denoiser source, old data/checkpoint provenance and 28 reused reference attempts before any new GPU update. The explicit stages are `prepare`, `verify`, `feasibility`, `train` and `evaluate`. Preparation and CPU checks do not construct the pretrained robot policy. A successful single-update gate must restore original weights, module modes, gradient flags and CPU/CUDA RNG, and reproduce the original first four supervised losses, before the fixed training run proceeds. These preparation checks establish implementation readiness; model performance requires the prescribed fourteen candidate attempts.

### Feasibility and training completed

The [single-update GPU gate](../results/robotics-preservation-pilot/feasibility.json) passes. Initial preservation loss is exactly zero; after one real update it is `2.4385e-6`, so the teacher exposes actual student drift. The four supervised losses reproduce the original training record exactly. The original student weights, frozen state, module modes, gradient flags and CPU/CUDA RNG are restored. The measured gate takes 3.99 seconds after loading/preparation and peaks at **1,111.29 MiB** allocated GPU memory. No action generation or pushing occurs in this gate.

The prescribed [training run](../results/robotics-preservation-pilot/training.json) then completes **600 updates**, with **3,000 student denoiser training forwards and 600 frozen-teacher final-block forwards**. It takes **280.28 seconds (4.67 minutes)** after policy loading, including validation/checkpoint work, and peaks at **1,116.30 MiB** allocated GPU memory. Validation additionally uses 512 student denoiser and 128 teacher-head forwards; those are outside the training-forward counts. All **2,400 original expert/recovery sample slots** remain unchanged, with 600 separate baseline anchoring windows. The learned six-tensor head is saved separately, with SHA-256 `d644fad4edf7cbedc50f31830625ba48b38f1ac5950bc02d627bd64d5f00eb3b`.

The [independent training audit](../results/robotics-preservation-pilot/training-audit.json) passes: the original first supervised losses match, all sample groups/splits and objective sums match the fixed recipe, teacher/backbone digests remain original, and only the same **1,313,282 final-block parameters** change. The audit reads saved records and small CPU head tensors; it does not repeat gradients.

On the fixed validation windows, final expert/recovery denoising losses are 0.003035/0.016870, compared with 0.004444/0.017235 for the established recovery-trained head on the same windows/noise. The new candidate's baseline denoising loss is 0.006500, versus 0.005654 at initialization, and its mean teacher-output discrepancy is 0.000828. These are supporting training diagnostics, not pushing success or checkpoint-selection criteria. The completed comparison below judges the final-update-600 checkpoint without selecting another training step.

### Completed output-preservation comparison

All **fourteen new candidate attempts** finish in one invocation. The **28 original/current reference attempts** are reused unchanged, giving 42 scored comparisons on the same seven development scenes and two prediction draws. No completed attempt is replaced or excluded. Four new startup checks confirm exact original-action restoration. All methods retain the native observations/actions, H16 generation, 100 denoising steps, 300-movement limit and strict >95% success criterion.

The [completed summary](../results/robotics-preservation-pilot/summary.json) reports:

| Method | Successful pushes | Average final alignment | Average predictions | Average recorded runtime* |
|---|---:|---:|---:|---:|
| Original / fixed eight, reused | 11/14 (78.6%) | 91.60% | 25.93 | 127.30 s |
| Current recovery-trained / adaptive, reused | 11/14 (78.6%) | 93.21% | 24.00 | 128.35 s |
| Output-preservation / adaptive, new | 10/14 (71.4%) | 94.82% | 24.50 | 142.32 s |

*Reference timings come from an earlier invocation. The architecture, H16 shapes, 100-step inference and maximum allocated GPU memory are unchanged, while episode-weighted average prediction times differ: 5.2730 seconds for the new candidate versus 4.8881 for current and 4.8746 for original. These measurements do not isolate an algorithmic speed effect. The time difference cannot be attributed only to the extra 0.50 predictions per attempt; session-level effects remain unresolved. Runtime is descriptive across sessions.

The candidate succeeds **5/7 in each prediction draw**; each reference succeeds 5/7 at offset 91000 and 6/7 at offset 92000.

| Method | Baseline holdouts / 8 | Recovery validation / 6 | Gains / losses versus original fixed eight |
|---|---:|---:|---:|
| Original / fixed eight | 6 | 5 | Reference |
| Current / adaptive | 8 | 3 | 2 gained / 2 lost |
| Output-preservation / adaptive | 5 | 5 | 3 gained / 4 lost |

Compared with current, the new candidate repairs **15/91000, 16/91000 and 16/92000**, succeeding at movements **225, 192 and 155** respectively. These are all three current-method failures in this development screen. It also loses four current successes; all four time out at movement 300:

| New lost scene / draw | Current success at movement | Candidate final alignment |
|---|---:|---:|
| 1 / 91000 | 195 | 93.85% |
| 5 / 91000 | 294 | 91.40% |
| 15 / 92000 | 168 | 92.74% |
| 23 / 92000 | 165 | 89.17% |

Both development baseline losses of the current method, 15/91000 and 16/92000, are repaired, but four other original-baseline successes are lost. Thus selected-case repair does not establish reduced negative flips overall. Scene 15 itself illustrates the trade: the successful prediction draw switches between checkpoints, leaving one success out of two for that scene. For every development attempt, at least one of the current and output-preservation checkpoints succeeds; this is diagnostic complementarity, not an executable 14/14 policy or permission to select models using eventual outcomes.

The predeclared development gate **fails**: original-baseline losses are four, above the required fewer-than-two, and overall success is ten, below eleven. Recovery-validation success meets its separate requirement. The candidate is **not adopted**. The forty confirmation scenes and their nine original losses were not rerun or used as training/selection examples.

| Output-preservation minus current | Observed difference | Paired-scene bootstrap 95% interval |
|---|---:|---:|
| Success rate | -7.14 percentage points | -35.71 to +28.57 pp |
| Final alignment | +1.61 percentage points | -1.61 to +5.29 pp |
| Predictions per attempt | +0.50 | -5.43 to +5.43 |
| Recorded runtime | +13.97 s | -17.62 to +40.38 s |

These descriptive intervals average both draw differences within each of seven scenes before 10,000 bootstrap resamples. All include zero; the cohort is small, previously inspected development data, and one training seed/penalty weight was tested. The result establishes neither general superiority nor a unique explanation of each failure. Better denoising validation loss and higher average alignment do not substitute for the failed retention gate.

### Collection cost and final checks

The [evaluation resource record](../results/robotics-preservation-pilot/evaluation-resources-from-000-1794093436907742200.json) covers **34.31 minutes**, after initial CPU input verification and including model/head loading, startup checks, all fourteen attempts, final checks and restoration. Together with the measured 4.67-minute training phase, these intervals total approximately **38.98 minutes**; model loading/preparation outside the training timer and implementation time remain separate. Evaluation peaks at **2.48 GiB sampled process RSS**, including loading, with at least 3.93 GiB available system memory. Its sampling interval includes loading, unlike the first rehearsal collector's interval, so those process-RAM peaks are not directly comparable. Maximum allocated GPU memory during attempts is **1,095.41 MiB in every arm**; no teacher is deployed.

The [independent complete audit](../results/robotics-preservation-pilot/artifact-audit.json) passes for **14 new scored attempts, 28 reused references, four new startup checks and 9,013 recorded movements**. It verifies model identities, unchanged frozen inputs/training provenance, exact original restoration, common initial physical/RGB states, action-queue indices, H16 accounting, first-success/timeout arithmetic, all means and gained/lost lists, and scene-cluster statistics. It performs no new neural inference, gradients or physics. The collector separately verifies the original weights and unchanged backbone at completion. All new source, tests, protocol, heads and evidence stay frozen on the experiment branch.

Recheck this completed pilot without new training or pushing:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_preservation_pilot.py --stage verify
.\.venv-robotics\Scripts\python.exe experiments\robotics_preservation_pilot_audit.py --stage complete
```

## 6. Further literature and implications for this implementation

The literature offers different preservation mechanisms; their relevance depends on whether a loss comes from changed model outputs or from the states/execution decisions encountered in the feedback loop.

| Primary source | Preservation idea | Relevance and limit here |
|---|---|---|
| [L2-SP, ICML 2018](https://proceedings.mlr.press/v80/li18a.html) | Penalize movement away from pretrained parameters. | A parameter penalty could anchor the final block during training. Parameter proximity alone does not ensure the same closed-loop outcome. This is distinct from the current output penalty. |
| [WiSE-FT, CVPR 2022](https://arxiv.org/abs/2109.01903) | Blend pretrained and fine-tuned weights into a single checkpoint. | Our six same-shaped final-block tensors could be blended offline, keeping the deployed architecture unchanged. The paper studies classification robustness; PushT retention remains untested. Our nonlinear final block also means weight blending is not generally identical to blending predicted actions. |
| [Model soups, ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html) | Average compatible checkpoints fine-tuned from a common pretrained model. | This motivates combining the current and output-preservation heads, which have complementary development outcomes and the same frozen backbone. Its classification results do not guarantee combining their robot-control successes. |
| [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html) | Obtain expert labels at states visited by the learner and aggregate them into training. | This addresses the effect of actions on future observations. It motivates checking coverage of the states visited under adaptive execution; it would require trusted new corrective labels rather than treating simulator scores as action labels. |

The saved dataset contains **six recovery episodes from the failure-pattern investigation and twenty-five from the benefit-aware pilot**. For example, scene 2's `recovery-000`/`recovery-001` metadata records a branch after movement 157 from an original failure, followed by cached continuation or one-shot replanning. These are useful successful continuations, but do not establish coverage of every state later visited by a changed checkpoint with adaptive execution. This is an implementation-grounded reason to keep closed-loop distribution shift separate from forgetting; it does not establish the cause of any particular loss.

These are literature-informed directions, not additional implemented candidates. The approved weight-1.0 output-preservation run and its fourteen-attempt schedule stay frozen. The existing result that seven of nine confirmation losses succeed with trained/fixed-eight execution also cautions against expecting a weight-only preservation mechanism to eliminate every execution-associated loss.

**Execution-side work:** [PACE (2026), sections 3 and 6](https://arxiv.org/html/2606.00537v4) chooses replanning boundaries from smoothed speed valleys in one predicted chunk, with a threshold calibrated from training demonstrations and no extra chunk samples. Its authors explicitly limit the method when the base policy keeps generating incomplete or incorrect actions. That limitation is relevant to our recorded stationary tails despite repeated predictions. Its default 50-command joint-space setting differs from our short H16, absolute two-dimensional commands; a PushT adaptation would require separate calibration and an agreed execution budget. This is a diagnostic hypothesis, not demonstrated negative-flip recovery.

[DEHP (2026)](https://arxiv.org/abs/2606.11408) instead trains a lightweight horizon-selection branch with online RL while freezing the chunk policy. [STEP (ICML 2026)](https://proceedings.mlr.press/v306/li26aa.html) uses consistency-based warm starts for denoising acceleration. They motivate horizon selection and temporal coherence, but would introduce additional training or inference components beyond this pilot. None of these sources establishes that the original baseline's successful PushT cases will all be retained by our controller.

## 7. Proposed follow-up: one fixed checkpoint average

**Reviewable proposal only; no averaged checkpoint, new implementation or new evaluation has been created.** The two trained checkpoints solve complementary development attempts. [Model soups](https://proceedings.mlr.press/v162/wortsman22a.html) provides a reason to test whether averaging their compatible weights can preserve useful behavior in a single deployed model. This is an adaptation to our policy, not evidence that the paper already solves PushT negative flips.

1. Use the established recovery-trained final600 head and the new output-preservation final600 head as the two endpoints. Their frozen backbone/normalization are identical and their six FP32 parameter tensors have matching names and shapes. Form exactly **one 50/50 midpoint**, computing each element's mean in float64 and casting once to float32. Preserve both endpoint files and bind their hashes. No retraining, coefficient search, per-scene selection or online multi-model inference is included.
2. Use separate new source, protocol, checkpoint and result paths, with CPU checks of tensor arithmetic, finite values, unchanged endpoints/backbone, protected evidence and resumption. The deployed model retains the existing architecture, H16 generation, two observations, 100-step sampler and unchanged adaptive controller. A weight average of the nonlinear final block need not reproduce an average of its action outputs; its behavior must be measured.
3. Run the original/current/midpoint/original short startup controls, then **fourteen midpoint attempts** on the same seven development scenes and offsets 91000/92000. Reuse the 28 original/current references and the fourteen pure preservation attempts as supporting comparisons. Complete the new fourteen regardless of intermediate results. Keep the 300-movement limit and strict >95% scoring.
4. Report all gains/losses, overall/recovery success, alignment, predictions and descriptive runtime. Retain the existing development screen: fewer than two original-baseline losses, at least 11/14 successes and at least 3/6 recovery successes. Report any newly lost current successes as well. A favorable screen would justify discussing a fresh confirmation, not automatic adoption. All these seven scenes remain inspected development data; scenes 170–209 stay excluded from training/selection.

There is no model training in this proposal. The last candidate's fourteen-attempt collector took 34.31 minutes including loading/checks; a reasonable execution estimate for this fixed test is **30–45 minutes plus implementation time**. It tests whether one averaged model can retain the complementary repairs without sacrificing existing successes. The failure sets alone do not prove it will succeed.

## Reproduction and validation

From the project root, diagnostics and the independent audit use the existing robotics environment and standard-library code:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_baseline_loss_review.py
.\.venv-robotics\Scripts\python.exe experiments\robotics_baseline_loss_review_audit.py
.\.venv-robotics\Scripts\python.exe -m unittest discover -s tests -p test_robotics_baseline_loss_review.py
```

The **15 CPU tests pass**. They cover within-window drawdown versus net progress, early-ended cached suffixes, exact movement indices, RGB/action prefix mismatches, protected evidence, all nine losses, successful controls and component outcome counts. The independent audit passes and verifies **724 source/evidence bindings**. Existing results and frozen sources remain unchanged.

The optional coverage plot uses `--plot` with Matplotlib available. It was rendered with isolated scratch plotting packages (Matplotlib 3.9.2, NumPy 1.26.4); no plotting package was installed into `.venv-robotics`. Numerical review and verification require no new plotting dependency.

The saved-trajectory diagnosis is retrospective, and both preservation pilots are development evidence. Together they identify two unsuccessful retention recipes and complementary recovery behavior, not a validated negative-flip fix or a generalization result. The [earlier loss review](2026-10-13-robotics-lost-success-review.md) and its stopped controller directions remain preserved.
