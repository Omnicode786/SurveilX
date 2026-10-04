# Detection uncertainty: accuracy-g3-weapons-yolo

Dataset: `dangerous-items-development-v1`. Test AP50: 0.5041.

95% cluster-bootstrap interval: 0.4048–0.6132 from 500 resamples across 106 recorded groups.

| Class | Point AP50 | 95% interval |
|---|---:|---:|
| firearm | 0.5278 | 0.3582–0.7215 |
| knife | 0.4453 | 0.2644–0.6112 |
| machete | 0.4341 | 0.2631–0.6169 |
| baseball_bat | 0.6091 | 0.4365–0.7743 |

A narrow interval cannot establish site/source independence when dataset provenance lacks those groups. This interval was not used for model selection.
