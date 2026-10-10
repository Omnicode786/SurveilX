# Non-training completion pass — 2026-10-05

## Completed

- Replaced order-dependent greedy tracking with class-aware global IoU assignment; preserves expiry and ID lifecycle. Optional SciPy acceleration has a tested NumPy fallback. Assignment changes can alter runtime track IDs; policy behavior is covered by the safe suite, but crossing/occlusion field performance remains unmeasured.
- Added a read-only labeled-sequence tracking evaluator, CLI, per-class identity metrics and synthetic correctness report. No existing real tracking labels were available.
- Added role-protected dataset readiness API and Learning UI. Existing files, hashes and splits are validated; classes missing from a split or with fewer than ten labels are visible. The ten-label warning is a heuristic, not statistical acceptance.
- Added generation checkpoint/stopping information and expandable epoch telemetry. Existing SVA histories and YOLO CSVs remain supported. YOLO losses retain their own box/classification/DFL names; no historical objectness/GIoU totals are invented. YOLO F1 is explicitly derived from recorded precision and recall.
- Corrected stale completion/phase/design records: all g4 campaigns completed, no training pending. SVA v2 remains implemented but untrained and accuracy-unvalidated.
- Restarted the local API with automatic training disabled, verified authenticated read-only endpoints and inspected the new UI. No model activated or accepted.

## Verification

192 non-training tests passed in the full suite, with the real scene training test deselected. Sixteen focused tests passed after the CSV telemetry addition, covering one additional test (193 distinct safe tests overall). The first focused run exposed an incorrect F1 expectation in a new fixture; corrected to 1/3 and rerun passed. Ruff passed. Frontend build passed (`index-CZzKPaTs.js`). Existing Starlette/ONNX warnings and expected unreachable-RTSP diagnostics remain.

Evidence: `reports/nontraining-product-api.json`, `reports/nontraining-final-integrity.json`, `reports/nontraining-sh17-development-v2-coverage.json`, `reports/nontraining-dangerous-items-development-v1-coverage.json`, `reports/nontraining-dataset-coverage-ui.png`, `reports/tracking-synthetic-evaluation.json`.

## Still incomplete

- Training and controlled accuracy validation of the implemented SVA revisions, only when explicitly authorized later.
- Independent reviewed footage for each requested domain; theft/collision supervision, signal/machine-state/relation labels and stronger sparse PPE coverage.
- Actual identity-labeled sequences, independent scene labels, operator trials, competing real-stream recall measurements and longitudinal drift/retention evidence. Software cannot manufacture these measurements from synthetic fixtures or inherited class names.
- Learned semantic environment inference and supervised contextual risk costs require those reviewed targets; current configured environment/site rules remain explicit.

Paper, complete accelerator deployment/hardware coverage and full production infrastructure/verification remain excluded. No new datasets were downloaded or deleted in this pass; previous proven cache cleanup is recorded separately in `reports/g4-cleanup-journal.json`.
