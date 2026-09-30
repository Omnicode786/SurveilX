# Evaluation protocol

## Executable checks

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check surveilx training tests scripts
.\.venv\Scripts\python scripts/benchmark.py
.\.venv\Scripts\python scripts/verify_live.py
.\.venv\Scripts\python -m training.evaluate data/datasets/generated-v2/manifest.json data/runs/new-suite --epochs 25
```

Use a new output directory each run. `verify_live.py` uses the generated local admin credential and modifies generated test incidents by acknowledging and labeling them. It is intended for this local test workspace, not a production incident store.

| Experiment | Executed scope | Required real evaluation |
|---|---|---|
| Baseline inference | HOG microbenchmark; synthetic fixture end-to-end; real Penn-Fudan scratch/YOLO AP50 and precision/recall comparison | Broader multi-domain detection and tracking ID metrics |
| SVA-Net training | Generated motion clips; independent train/validation/calibration/test files | Site-specific interactions, events, rare critical cases and expert annotations |
| Calibration | Temperature fit, Brier/ECE/risk–coverage | Site/shift calibration with adequate sample counts and uncertainty intervals |
| Architecture ablations | Full, no motion, no context, no interactions, no temporal convolution | Multi-seed real-domain comparisons against generic temporal baselines |
| Scheduling | Deterministic four-camera simulated costs; FIFO, round robin, static priority, greedy priority, MCRS | Measured heterogeneous model costs and simultaneous incidents |
| Hardware | Current CPU, RAM and available NVIDIA telemetry | CUDA/TensorRT/FPGA/Jetson/Pi measurements on actual targets |
| Human loop | API/UI acknowledgment, labels and append-only application audit | Operator workload, missed reviews, inter-rater reliability and escalation accuracy |
| Reliability | Input validation, role gates, terminal-state transitions and source frame progression | Fault injection against PostgreSQL, object store, actual RTSP and external providers |

## Ablation questions

- No motion: do trajectory and feature differences improve motion-event discrimination?
- No context: does context help under real domain transfer? Constant synthetic context cannot answer this question.
- No interactions: do pairwise messages improve relational events beyond independently pooled entities?
- No temporal convolution: does temporal filtering help beyond averaged motion features?
- Always detector versus adaptive: does compute reduction preserve event recall and coverage? Requires labeled replay, not only scheduler simulation.
- Raw versus calibrated confidence: does calibration improve held-out NLL/Brier and selective error without hiding poor accuracy?

Do not tune architecture against the final test set. Current generated runs are exploratory; after changing the model in response to observed results, use a genuinely untouched dataset for final acceptance. Report all variants, including failures and calibration regressions. Synthetic perfect scores, if any, are not surveillance accuracy claims.

## Failure categories

Record source failure/reconnect, dark/blurred input, camera motion, occlusion, small entities, invalid annotations, domain mismatch, missing expert, expert exception, thermal/memory pressure, missed coverage, database write failure, spool replay, evidence expiration, notification failure, unreviewed feedback and model checksum mismatch. Runtime exceptions currently report error type without secrets; richer structured traces and production fault-injection remain open.

## Reproducibility

Each completed run saves the dataset hash, model hash, seed, epochs, architecture flags, split counts, runtime versions, selected device, optional Git commit and duration. The workspace began without Git history; early run commit fields are null. `requirements-lock.txt` records the current Windows CPU environment and is not a portable CUDA/Jetson lock. Preserve original files under `data/runs` and generated dataset versions. Full vendor-runtime manifests and energy measurements need target hardware.

## Reviewed adaptation and acceptance verification

On 2026-09-28, the full suite passed 47 tests. The new coverage includes independent annotation review, train-only evidence export, unchanged held-out splits, duplicate-evidence rejection, replay, automatic-worker lifecycle, acceptance leakage checks and artifact-bound approval. Two focused export regressions also passed after replacing nested replay paths with content-addressed asset names. The frontend build and browser checks passed.

`reports/acceptance-workflow.json` records a frozen event-model evaluation on 100 test clips from a separate synthetic seed: accuracy 1.0, nominal Wilson lower bound 0.9630, and ECE 0.0. These easy generated labels verify software behavior, not surveillance accuracy. The numeric gate passed, but production eligibility remained false and an approval request returned HTTP 409.

`reports/acceptance-detector-smoke.json` verifies the new evaluator with scratch and adapted-YOLO artifacts. It reproduces their earlier Penn-Fudan AP50 values (0.624879 and 0.864502) and rejects the same dataset as independent acceptance because its source groups overlap training lineage. This is an inference integration check, not new real-domain acceptance evidence.

See [reviewed learning](reviewed-learning.md) for the operational workflow and remaining limits.
