# Detection choices and processing cost

Each camera's saved detection choices control model execution as well as displayed results. No model is trained, accepted or activated by these choices.

- Disabled categories are removed before route allocation. Shared experts run once per frame for the union of their enabled routes; disabled routes cannot consume the bounded model budget. Fallback/retry behavior applies to enabled routes.
- Fully disabled detection and event specialists leave the service rotation. Disabled fallback models also skip their forward pass.
- Baseline and adapted YOLO pass explicit enabled class IDs to the installed Ultralytics predictor, before NMS and CPU detection conversion. Every call supplies `classes`, including `None` for an unfiltered camera, so filters cannot leak between cameras.
- Event histories are JPEG-encoded only when a compatible enabled temporal model needs them. Entity and scene histories are maintained separately. Object filtering, probability masking without renormalization, and PPE absence-rule dependency guards remain in effect.
- With no requested compatible model work, the camera does not reserve inference-scheduler budget. Direct inference calls also skip forwarding, tracking, history encoding, policies, evidence and per-frame decision writes. The source and preview remain available.
- Camera status includes `selection_work` for diagnostics: whether the primary model was skipped, enabled specialist/event counts, and whether temporal history was encoded. These are work-selection indicators, not GPU utilization or accuracy measurements.

Unknown legacy model taxonomies stay enabled conservatively. Already loaded models remain resident for other cameras and fast re-enabling; this feature saves execution, not necessarily VRAM. Models with selected classes still compute their shared backbone/head. It does not prune weights, lower resolution, change thresholds, or infer alternative labels for disabled winning classes.

## Inference-only measurement, 2026-10-24

NVIDIA MX130, CUDA PyTorch 2.14.0+cu126; preserved general YOLO11n checkpoint SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. Four existing Penn-Fudan validation images, 320 px, 20 iterations per setting, rotating setting order after warmup. CUDA synchronization brackets each call. A hook on the actual prediction backend verifies network forwards.

| Selection | Median call | p95 call | Actual backend forwards |
|---|---:|---:|---:|
| All 80 classes | 41.57 ms | 52.32 ms | 20 |
| Person only | 43.40 ms | 52.06 ms | 20 |
| All classes off | 0.745 ms | 0.823 ms | 0 |

Person-only showed no measurable speed benefit in this bounded sample. All-off avoided every GPU forward. Whole-specialist/event elimination and reduced downstream tracking/storage provide the meaningful savings; selecting fewer classes cannot promise proportional speed gains. These measurements do not establish accuracy, live-camera FPS, power draw or sustained throughput.

Raw samples, image paths and hashes: ignored local `data/testing/detection-choices-20261024/selection-benchmark.json`. Reproduce without training or deployment:

```powershell
.venv-cuda/Scripts/python.exe -m scripts.benchmark_detection_selection --weights data/runs/yolo11n-general-building-v1/weights.pt --dataset data/datasets/pennfudan/manifest.json --output data/testing/detection-choices-20261024/selection-benchmark.json
```

Regression tests cover pre-NMS IDs, all-off forward elimination, resetting filters between cameras, enabled route unions, budget release, enabled fallback recovery, disabled fallback suppression, event-only operation, no unused JPEG histories, and preservation of preview frames. Old `infer(frame, resolution)` callers remain compatible. Historical artifacts are unchanged; future training plans must verify changed source fingerprints instead of blindly resuming a frozen campaign.
