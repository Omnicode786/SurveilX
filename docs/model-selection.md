# Task-specific model selection

Implemented 2026-10-15. Recommendations use development validation evidence; they do not grant acceptance or activate models. Open **Model performance** in the command center, or use authenticated `GET /api/model-performance`.

## Evidence and comparison boundary

The catalog verifies completed checkpoint checksums and binds each evaluation sidecar to the checkpoint, model manifest, dataset SHA256, class ordering, task and validation split. Missing, uncalibrated, corrupted or incompatible evidence is excluded from rankings. Partial runs appear separately. Historical test results are displayed but never rank candidates.

Comparison groups require identical task, domain, synthetic/real status, dataset SHA256, taxonomy, representation and temporal input contract. D-Fire v1/v2 and SH17 v1/v2 remain separate. A person detector evaluated on Penn-Fudan cannot displace an SH17 person specialist on that evidence alone. Scene-clip event classification and object detection are separate tasks. Model families are not compared by parameter count or architecture name.

For detection, the primary metric is per-class AP50; for events it is per-class F1. Ordinary classes score `0.55 * primary + 0.25 * recall + 0.20 * F1`. Firearm, knife, machete, fire, smoke, fall and fighting use `0.40 * primary + 0.40 * recall + 0.20 * F1`, prioritizing missed critical events. A bounded `0.05 * ECE` penalty accounts for measured calibration. Available terms are normalized; missing measurements stay explicit. These are transparent engineering weights, not an empirically optimized safety utility or an acceptance criterion.

Warm native-profile latency only breaks equal-score ties when all tied candidates have comparable device measurements. Cold load/file-I/O times are never treated as runtime latency. CPU and GPU timings cannot break the same tie. Resource requirements and measured latency remain visible; incompatible devices are still rejected by the existing loader. Small validation sets and source dependence remain limitations even for a high score.

## Runtime behavior

Only experts already loaded through explicit deployment enter automatic routing. The catalog does not load recommended candidates into production, create canaries or alter deployment records. Existing acceptance, canary, rollback and restart restoration workflows remain authoritative.

For compatible loaded detection experts, the router:

1. Selects a model for each observed category using fingerprinted validation evidence.
2. Groups classes by model and calls each selected model once per camera frame, using its native calibrated geometry.
3. Keeps only that model's owned classes, preserving the existing class-aware tracker and incident pipeline.
4. Limits work to three models per frame and rotates the starting category when constrained; deferred categories are reported, not silently declared evaluated.
5. On failure, tries the next compatible loaded expert. Failed models receive bounded exponential cooldown and recover automatically. If every route fails, the explicit HOG/demo baseline remains available.

Successful expert fallback preserves the router's tracker identity. Falling all the way back to the baseline resets its tracking context. Named event experts are ranked only inside the same temporal contract; failure tries another explicitly loaded compatible event expert with a recovery cooldown. Detection output survives event-model failure. Different event corpora retain explicit scheduling rather than an invalid cross-task ranking.

Missing evidence or conflicting dataset contracts retain the prior named-slot scheduler. This conservative behavior avoids pretending that incomparable models have a validated shared ranking. Existing capture/inference separation, global camera scheduling, coverage deadlines, incident logging and camera management remain in place.

Each camera snapshot/evidence observation records category routes and component versions. Routed output is labeled `component_calibrated; pooled_reliability_unvalidated`: calibration of individual experts does not prove calibration of the combined stream.

## Reproducible tools

Run expensive inference diagnostics serially with training, or use the CPU environment while leaving CUDA to the active campaign. Do not launch a second GPU trainer.

```powershell
# Reuse valid reports and evaluate newly completed checkpoints on validation only.
.\.venv\Scripts\python -m scripts.evaluate_model_catalog

# Warm benchmarks require a free accelerator; do not overlap active CUDA training.
.\.venv-cuda\Scripts\python -m scripts.benchmark_catalog_runtime --versions cuda-g4-weapons-yolo

# Controlled transformed-real-image probes, not independent night/blur evidence.
.\.venv\Scripts\python -m scripts.evaluate_conditions cuda-g4-weapons-yolo

# CPU execution of the actual router; choose a fresh report path.
.\.venv\Scripts\python -m scripts.evaluate_routed_detector dangerous-items-development-v1 accuracy-g3-weapons-yolo cuda-g4-weapons-yolo --output data/routing-evaluations/new-comparison.json
```

Reports live under ignored `data/model-evaluations` and `data/routing-evaluations`; a source-only Git clone does not include models or data. Catalog evaluations reuse identical weights/data/calibration/geometry instead of repeating identical inference. Dashboard caches are bounded and invalidated by file size/mtime; runtime loading and acceptance independently verify integrity.

For this campaign, a read-only CPU finalizer is already waiting for both g7 and its bounded retry to finish. It populates fingerprinted validation reports for new completed models, after training process exit and available-RAM checks; the dashboard then recomputes recommendations automatically. It never trains, activates or promotes anything, and stops if a named campaign is stopped by the user. State/log: `data/model-evaluations/finalizer.json` and `.log`. For a future explicitly authorized campaign, run `python -m scripts.finalize_model_evaluations PATH_TO_CAMPAIGN_STATUS [OTHER_STATUS_OR_QUEUE_JSON]` using the CPU environment. Do not launch a duplicate worker over the same output lock.

## Remaining validation

Independent reviewed streams are still required for pooled calibration, false alarms per camera-hour, critical-event delay and site acceptance. Real identity trajectories are required for tracking measurements including HOTA. Photometric probes preserve labels but do not create independent lighting/blur/occlusion test data. No learned production deployment was created in this pass.

See [measured outcomes](selection-results-1015.md), [training status](training-status.md), [g4 diagnosis](g4-sva-diagnosis.md) and [SVA v3 design](sva-v3-design.md).
