# Visual SAC pilot on Gym-PushT

**Visual SAC ran comfortably on the GTX 1650, but this short run did not produce a useful pushing policy.** It stopped at the planned ten-minute training cap after 16,589 movements and 15,588 updates. It completed 0/8 development pushes, and average alignment fell versus the untrained actor. The result verifies hardware feasibility for this configuration. The existing Diffusion Policy comparisons remain the scored project evidence.

## Matched task and pilot settings

- **Policy:** [Stable-Baselines3 2.7.1 SAC](https://github.com/DLR-RM/stable-baselines3/tree/v2.7.1/stable_baselines3/sac), `MultiInputPolicy` with its standard image encoder and a 256/256 MLP. SAC starts from random weights, trains online with no demonstrations, and emits one x–y target per movement. It is an adaptation of the [SAC algorithm](https://proceedings.mlr.press/v80/haarnoja18b.html), which did not evaluate PushT.
- **Inputs:** two consecutive native 96 × 96 RGB observations plus the corresponding two pusher positions. The initial observation is duplicated at reset. The wrapper stacks image channels, scales pusher coordinates by 512, and passes the original action space, reward and termination through unchanged. Exact block pose and coverage are never policy inputs. Native coverage-based reward is available to the learner during training; true coverage is also used for scoring.
- **Environment:** same Gym-PushT version, 300-movement limit, and strict coverage greater than 95% for success. Training draws initial scenes from `[100000, 1000000)` with a fixed seed, disjoint from the project’s earlier scene panels. Deterministic before/after diagnostics use the same eight previously inspected scenes 0–7. No new Diffusion Policy evaluation was run.
- **Budget:** 20,000 transitions of replay capacity, batch size 64, one update per movement after warmup. The hardware preflight allows 256 movements; the separate from-scratch pilot stops at 20,000 movements or ten minutes of training, whichever comes first. Guard checks sample system RAM and free GPU memory and read PyTorch’s peak allocation counters. These choices were fixed before looking at pilot outcomes.

## Results and limits

| Policy on development scenes 0–7 | Successes | Mean final coverage | Mean peak coverage |
|---|---:|---:|---:|
| Untrained visual SAC | 0/8 | 12.58% | 13.93% |
| Visual SAC after 16,589 movements | 0/8 | 8.30% | 9.92% |
| Historical fixed-eight Diffusion Policy | 3/8 | 93.43% | — |

The historical row reuses the [fixed-eight development records](../results/robotics-adaptive-length/episodes.json), with one recorded prediction draw per scene. Both SAC evaluations matched all eight starting coverage values exactly. The comparison is descriptive: it uses inspected development scenes, one short SAC training seed, and policies with different training histories. It does not establish that a well-trained visual SAC policy would fail, or that Diffusion Policy generally outperforms RL. It also does not justify presenting this checkpoint as the project’s RL comparison baseline. A [later PushT-Vision study](https://arxiv.org/html/2610.18677v1#A6) also reports difficulty training SAC, but uses different actions, rewards and scoring; its results cannot be pooled with ours.

The [preflight](../results/robotics-visual-sac-pilot/preflight/summary.json) completed 192 updates in 9.44 seconds. In both the preflight and [pilot](../results/robotics-visual-sac-pilot/pilot/summary.json), actor and critic weights changed and the final losses and parameters were finite. In the pilot, peak PyTorch GPU allocation/reservation were 124.44/144.00 MiB, peak sampled process RAM was 2.84 GiB, and minimum sampled available system RAM was 5.22 GiB. The full replay arrays have a capacity of 2.06 GiB; the time limit was reached before the buffer filled. The 36.29 MiB saved checkpoint excludes replay. Training stopped because of the time cap, not memory pressure.

Both CPU observation/simulator checks passed. A separate saved-artifact check reproduced the aggregate scores, checked the success threshold and scene matches, and verified source and checkpoint hashes. The saved actor reloaded on CPU and returned a valid target. Preserve these results and discuss a larger budget or a different visual RL baseline before another run.

## Reproduction

Use the existing `.venv-robotics`. Add the pinned pilot dependencies to the separate ignored directory; do not replace the established CUDA or Gym-PushT packages. Choose unused run IDs to preserve existing results.

```powershell
.\.venv-robotics\Scripts\python.exe -m pip install --no-deps --target tmp\visual-sac-deps -r requirements-visual-sac.txt
.\.venv-robotics\Scripts\python.exe -B -m unittest discover -s tests -p test_robotics_visual_sac.py -v
.\.venv-robotics\Scripts\python.exe -B experiments\robotics_visual_sac.py --stage preflight --run-id preflight-repeat
.\.venv-robotics\Scripts\python.exe -B experiments\robotics_visual_sac.py --stage pilot --run-id pilot-repeat --preflight-id preflight-repeat
```

The [runner](../experiments/robotics_visual_sac.py) records the protocol, source hash, resource counters and per-scene development scores. A saved policy is local in `tmp/visual-sac-models/`; it does not include the replay buffer. Exact training trajectories and elapsed time may vary with CUDA and desktop load. The project’s earlier frozen checkpoints and evaluation files are kept separate.
