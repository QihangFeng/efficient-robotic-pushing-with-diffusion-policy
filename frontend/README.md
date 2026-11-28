# PushT experiment replay

The demo uses four recorded attempts from the [recovery confirmation evaluation](../results/robotics-recovery-confirmation/evaluation/). Each pair shares a starting scene and prediction draw.

| Scene | Draw offset | Original fixed-eight | Fine-tuned adaptive |
| --- | ---: | --- | --- |
| 173 | 60000 | Movement limit | Success |
| 177 | 70000 | Success | Movement limit |

## Replay data

From the repository root, regenerate or verify the checked-in browser data with Python 3.11. The exporter uses the standard library and reads the saved traces and attempt records.

```powershell
py -3.11 frontend/export_replays.py
py -3.11 frontend/export_replays.py --check
```

Frame zero is the starting scene. Each following frame records the state after one movement. Prediction markers identify the first movement using a new plan. Controller decisions are recorded after the movement that triggered them. Coverage is the simulator's recorded overlap with the target. Positions use PushT workspace units and angles are expressed in degrees.
