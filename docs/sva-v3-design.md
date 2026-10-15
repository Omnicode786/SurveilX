# SVA v3 — accuracy-oriented detection revision

Implemented 2026-10-13. **Current status, verified 2026-10-15:** weapons v3 completed with test AP50 0.081309; PPE was interrupted after epoch 19 with validation AP50 0.125835 and no final test report. Training is stopped by user request; no models accepted or activated. See [training status](training-status.md). The research and initial preparation notes below describe the architecture pass before the later g6 campaign.

## Evidence and research

G5 weapons classifier v1/v2 peaked at validation AP50 0.0567/0.0631, below the parent's 0.0775. PPE coverage-v2 removed lost supervision but peaked at 0.0851, below its parent0.0876. Training classification losses improved while validation classification worsened. Thus a tiny classification adapter and assignment repair were insufficient; backbone/features, score learning and generalization require a more substantial experiment. These observations do not isolate one causal architectural defect.

Primary sources consulted:

- [EfficientDet paper](https://arxiv.org/abs/1911.09070) and [Google Research explanation](https://research.google/blog/efficientdet-towards-scalable-and-efficient-object-detection/): efficient weighted bidirectional feature fusion. V3 implements a small independent four-level fusion block; it is not the complete EfficientDet/BiFPN architecture.
- [Generalized Focal Loss paper](https://arxiv.org/abs/2006.04388): continuous quality-aware class targets and confidence ranking. V3 uses dense quality-modulated BCE, with a documented bootstrap, while retaining SVA's box coder/GIoU. It does not implement GFL's distributional box regression or claim the paper's results.
- [VarifocalNet paper](https://arxiv.org/abs/2008.13367): joint localization/class confidence supports evaluating a quality score instead of relying on separately learned ranking signals. V3 does not copy its star-shaped feature representation or varifocal loss.
- [TOOD paper](https://arxiv.org/abs/2108.07755) and [author implementation](https://github.com/fcjian/TOOD): classification/localization alignment. Reviewed as a possible later assignment experiment; task-aligned dynamic assignment is **not implemented** in this pass.
- [FCOS paper](https://arxiv.org/abs/1904.01355): multiscale anchor-free detection context. V3 remains independently implemented and retains the existing SVA center/size coder.

## Actual changes

| Change | Purpose | Limit |
|---|---|---|
| Stride4/8/16/32 pyramid | Retain finer detail for small PPE/body-part objects | More activation memory and computation; low-information pixels and sparse data remain |
| Learned positive normalized top-down/bottom-up fusion | Exchange fine detail and coarse semantics | One compact fusion stage; no promised generalization gain |
| Shared semantic mixer and stronger decoupled classification tower | Improve features supplied to class and geometry branches | Requires end-to-end supervision; larger capacity can still overfit |
| 10% class-feature dropout | Add mild regularization | Must be evaluated on each domain; disabled in evaluation |
| Dense quality focal class supervision | Teach wrong classes/background suppression and localization-aware score ranking | New objective/score scale requires fresh calibration |
| Quality-class score used directly in decode | Align training target with inference ranking; auxiliary objectness is not multiplied into score | V1/v2 retain their original product scoring exactly |
| Four-level coverage assignment | Reserve supervision for overlaps/tiny objects, adapt scale mapping to finer pyramid | Uses the existing geometric matching design, not TOOD dynamic assignment |
| Six-epoch quality bootstrap | Interpolate positive targets from1 toward detached predicted IoU, preventing weak initial boxes from starving class learning | Reference validation always uses actual IoU; bootstrap is our adaptation, not a paper reproduction |
| Three-epoch LR warmup then cosine decay | Avoid abruptly updating newly initialized semantic/quality heads | No hyperparameter optimality claim |

QFL-style class objective sums BCE(logit,target) times (sigmoid(logit)-target)^2 over all classes/locations, normalized by the positive count (minimum1). Positive target is detached IoU, with training-only bootstrap q=b+(1-b)IoU; negatives target0. Loss combines classification +0.25 auxiliary objectness +3 GIoU +2 smooth-L1. Class-weight/focal legacy flags are rejected for v3. History labels the new objective, bootstrap and reference loss accurately.

Default width24/depth2 with17 classes has **1,619,004 parameters**, versus1,403,566 in v1 (about15.35% more). Small feature maps cost more despite the modest parameter increase. At the initial architecture pass, v3 CUDA memory had not yet been measured; the later measurements are recorded below. Normal computers remain supported through PyTorch CPU.

## Compatibility and initialization

`--architecture-version 3` selects the new version. Legacy v1/v2 defaults and tensor keys remain intact. V3 checkpoints require the new loader. Runtime/evaluator use the same decoder and shared stretch/letterbox preprocessing; new calibration must be fitted on calibration-only data.

Optional `--initialize-from LEGACY_RUN --transfer-from-legacy` validates exact v1/v2 weights, taxonomy/domain/capacity, then copies only stem, stages and old lateral features to their matching v3 levels. New fusion, semantics, class/quality and geometry heads remain initialized independently. Transfers are explicitly recorded by key; neither legacy logits nor calibration are reused. A transferred initialization is **not** a retained legacy model and is not entered as the parent-selection baseline; evaluate the separate frozen parent for comparison. Exact v3-to-v3 continuations retain ordinary checked full-checkpoint selection.

Classification-only v3 training is rejected because it would freeze the newly changed semantics/quality machinery. ONNX deployment fails explicitly for v3 until its four-level score contract is validated; existing three-level exports continue working. This is a compatibility guard, not complete acceleration work.

## Historical verification and initial experiment preparation

200 safe tests passed in the full suite, with the actual scene training test excluded. After runtime-parity and final integration additions,33 focused tests passed. V3 tests include finite forward/synthetic backward (no optimizer), four-scale assignment, tiny/overlapping/empty labels, background gradients, decoder score semantics, strict checkpoint round trips, explicit feature transfer, PyTorch runtime parity for stretch/letterbox, and rejection of legacy ONNX contracts. The first parity fixture referenced the wrong Detection field; corrected from confidence to score, rerun passed. Ruff and TypeScript/Vite build passed (`index-CqCW9NFi.js`). Existing ONNX/Starlette warnings remain.

`reports/sva-v3-structural.json` and `reports/sva-v3-compatibility.json` record structure, source hashes, actual frozen legacy output/loss parity, artifact integrity and unchanged g5 plan hash. No dataset downloaded, model accepted/activated or real v3 training started.

Prepared ONLY: `data/generations/sva-v3-prepared-1013/plan.json` contains weapons/PPE/fire v3 candidates from frozen g4 features, unchanged splits/resolutions/stretch geometry,32 maximum epochs, patience12, seed42, LR0.0003 and conservative batch2. It is **not queued or running**. First measure v3 CUDA memory with existing data after the active g5 GPU campaign finishes; lower batch to1 in a new immutable plan if required. Run weapons first; inspect validation AP50/per-class recall, dense-score/background errors and loss generalization before expensive broad continuations. Compare fire against its stronger control. Test results remain reports only; no activation or independent reliability claim.

The combined v3 revision changes several components, so it measures the bundle rather than proving which component causes a gain. Follow with single-component ablations only if the bundle improves validation. Severe PPE sparsity, missing theft/collision/state/relation labels and independent-site evidence remain data problems; architecture cannot manufacture those labels.

## Authorized restart and CUDA measurement (2026-10-13)

The user subsequently authorized stopping g5 and starting v3 plus the other supported model families. The earlier prepared-only notes above are historical. G5 stopped with eight completed jobs and one interrupted PPE YOLO job; partial files were preserved.

Measured v3 on the MX130 with temporary model copies, train-only SH17 samples and float32:512 pixels/batch1 reserved820MiB;384 pixels/batch2 reserved1000MiB.512 pixels/batch2 exceeded the75% process allocation ceiling. The new campaign therefore uses PPE batch1 and other scratch batches2. These short measurements establish correctness/memory on the sampled inputs, not a whole-run memory maximum or accuracy gain.

The replacement bundle runs four domain scratch candidates, four adapted YOLO continuations and four temporal candidates (two real-data domains, two explicitly synthetic checks). Existing splits are fixed. V3 new heads receive48-epoch budgets/patience16 (person32/12), with six-epoch quality bootstrap and three-epoch LR warmup. This tests the bundled architecture; component attribution requires later ablation. Sparse PPE and absent domain/event supervision remain unresolved data prerequisites.

Sources are copied and hashed per job, including the v3 core. Later source mutation prevents another queued launch rather than silently changing its recipe. Completed g5 fire YOLO is the new fire YOLO parent because its validation AP improved; its reported test AP slightly declined, and was not used to select that parent. No models are activated automatically.

## Stopped campaign outcome — verified 2026-10-15

The authorized g6 campaign subsequently completed weapons v3 (best validation0.149588 at epoch14; test0.081309 versus parent0.031503) and began PPE v3. Weapons stopped at epoch30 after patience16; PPE was stopped by the user after epoch19, with best validation0.125835 versus parent0.087623. PPE has no completed calibration/test result. Ten other jobs were cancelled. No training is currently authorized, and no model was accepted or activated. See [full results and preserved-artifact rules](training-status.md).
