# Mathematical foundations and verification boundaries

This is an implementation reference, not a research paper or proof of superiority. Equations below describe the current code. Hyperparameters and heuristics require empirical validation; established losses do not guarantee accuracy on arbitrary datasets.

## Data and task contract

Detection supervision is a set of normalized axis-aligned boxes and discrete classes:

\[
\mathcal D=\{(x_i,\{(b_{ij},y_{ij})\}_{j=1}^{n_i},c_i,z_i,g_i)\}_{i=1}^N.
\]

Here \(b=(x_1,y_1,x_2,y_2)\in[0,1]^4\) has positive area, \(y\in\{0,\ldots,K-1\}\), \(c\in\mathbb R^4\) is optional scene context, \(z\) is an optional zone raster, and \(g\) identifies a camera/session/source group. A group belongs to exactly one of train, validation, calibration or test. Content hashes catch exact duplicate files across splits; they do not prove that near-duplicates or undisclosed source overlap are absent.

COCO pixel top-left/width/height, YOLO normalized center/width/height and VOC one-based inclusive corners map into this common contract. Labels are mapped by explicit taxonomy; changing class order changes model semantics. Negative images have empty box and label lists. Crowd/ignore labels are rejected because ignoring their semantics would create incorrect background supervision.

Video events use \(X\in\mathbb R^{T\times3\times H\times W}\), persistent entity boxes \(B\in\mathbb R^{T\times N\times4}\), context and a clip-level categorical label. Each version declares a fixed input contract for batching. Current support is single-label classification, not multilabel temporal localization. See [dataset adapters](dataset-adapters.md).

## Scratch detection network

`training/detector_model.py` builds a independently initialized residual convolutional backbone and top-down feature pyramid with strides 8, 16 and 32. Width and depth are configurable. Feature-pyramid reuse follows an established pattern ([FPN](https://arxiv.org/abs/1612.03144)); the specific module combination is an engineering design, not a novelty claim.

For level features \(F_l\), the scene/zone transformation is:

\[
\widetilde F_l=F_l\odot[1+\tanh(A_lc)_s+\tanh(Z_l(z))]+(A_lc)_b.
\]

The affine and zone parameters start at zero, preserving identity at initialization. Shape/gradient tests verify that annotated nonzero context can train these paths. Penn-Fudan has no scene/zone supervision, so its neutral-context experiment does not validate their usefulness.

At each feature cell with normalized center \(p\) and step \(s\), predicted geometry is:

\[
\hat q=p+2s\odot\tanh(r_{xy}),\qquad
\hat w=4s\odot\operatorname{softplus}(r_{wh}),\qquad
\hat b=[\hat q-\hat w/2,\hat q+\hat w/2].
\]

Softplus guarantees positive width/height. Inference clips boxes to the image and rejects zero-area results. Scores are \(\sigma(o)\sigma(k)\), followed by class-aware NMS at IoU 0.5. This raw product is a ranking score, not a calibrated probability.

Assignment selects a scale from object size and a center neighborhood; overlapping assignments prefer smaller objects. These are heuristic choices. There is no guarantee that every heavily overlapping object receives a unique positive cell.

## Detection losses

For objectness, binary focal loss uses \(\alpha=0.25,\gamma=2\):

\[
L_{obj}=\frac1{\max(N_+,1)}\sum_j-\alpha_{t_j}(1-p_{t_j})^2\log p_{t_j}.
\]

This reduces contributions from easy background locations ([Focal Loss](https://arxiv.org/abs/1708.02002)). Positive cells receive binary class cross-entropy and geometry losses:

\[
L=L_{obj}+L_{cls}+3\,\overline{(1-\mathrm{GIoU})}+2\,L_{\mathrm{SmoothL1}}.
\]

\[
\mathrm{IoU}=\frac{|B\cap G|}{|B\cup G|},\qquad
\mathrm{GIoU}=\mathrm{IoU}-\frac{|C\setminus(B\cup G)|}{|C|},
\]

where \(C\) is the smallest enclosing box ([GIoU paper](https://arxiv.org/abs/1902.09630)). The coefficients 3 and 2 are implementation defaults, not mathematically optimal constants. AdamW, gradient clipping and cosine learning-rate decay optimize the training objective; validation AP50 selects the checkpoint. Empty-image and tiny-object tests check finite gradients.

## Video-event network

`training/models.py` samples shared visual features at persistent entity centers, adds temporal feature differences and box-coordinate differences, applies a depthwise temporal convolution, and computes pairwise relations from entity features and relative geometry. Temporal and entity means feed a categorical event head. Training minimizes:

\[
L_{event}=-\frac1M\sum_i\log\operatorname{softmax}(f_\theta(X_i,B_i,c_i))_{y_i}.
\]

The current training labels supervise only the event head and its upstream representation. State/relation output heads are not claimed as trained recognizers. Missing tracks cause runtime abstention; there is no missing-entity mask implementation. Synthetic motion labels only establish software behavior.

## Modified YOLO

`training/yolo_model.py` wraps three YOLO11 backbone blocks with identity-initialized channel, spatial and scene-affine residual adapters. The inherited detector retains its original training losses and pretrained parameters. This is a modified pretrained baseline, not a scratch network. YOLO training currently uses neutral context; trained risk semantics and context-aware batching remain unimplemented. Source and model licensing remains subject to [Ultralytics terms](https://www.ultralytics.com/license).

## Calibration, thresholds and abstention

Event temperature scaling fits \(T>0\) on the calibration split by minimizing categorical negative log-likelihood:

\[
p_k=\operatorname{softmax}(z/T)_k.
\]

This follows [temperature scaling](https://arxiv.org/abs/1706.04599). Detection calibration instead labels each score-ordered prediction correct only when it matches an unused same-class ground-truth box at IoU >=0.5, and fits:

\[
q(s)=\sigma(\operatorname{logit}(s)/T+b).
\]

The bounded grid minimizes binary negative log-likelihood. A score threshold maximizes calibration-set F1; the test set is not used to choose it. Positive temperature preserves ranking and therefore AP50 except numerical ties. Calibration is refused without sufficient correct/incorrect predictions. It estimates detection correctness in the sampled domain, not threat probability or a universal accuracy guarantee. Small calibration sets can overfit; independent-domain reliability remains necessary.

Arbitration compares only calibrated evidence with the same task, taxonomy and domain. Large componentwise probability disagreement requests review; missing or incompatible evidence abstains. A mean of individually calibrated probabilities need not remain calibrated, so pooled calibration is not assumed.

## Evaluation and scheduling

Detection AP50 is mean 101-point interpolated precision over recall at IoU 0.5, with score-ordered, one-to-one, class-aware matching. Precision/recall/F1 use the calibration-selected operating threshold. This is not COCO AP averaged across IoU 0.50:0.95. False-positive/false-negative counts are retained alongside averages.

The global scheduler is a deadline-first heuristic with bounded service credit:

\[
C_{t+1}=\min(C_{max},C_t+B_t)-\sum_{i\in S_t}\hat c_i,\qquad
C_{max}=B_t\max(1,\lceil D/\Delta\rceil).
\]

Selected predicted costs fit available credit. Overdue cameras can reserve accumulated credit; a cost exceeding the coverage-window capacity is explicitly marked infeasible. Cost estimates, overload and non-preemptible inference mean this is not a hard real-time deadline guarantee or a proven optimal scheduler. The implementation is in `surveilx/controller.py`; tests cover reservation and infeasible costs.

## Reviewed replay and independent acceptance

Independent acceptance uses the lower endpoint of a two-sided nominal 95% Wilson score interval. For \(k\) successes from \(n>0\) trials, \(\hat p=k/n\), \(z=1.96\):

\[
L=\frac{\hat p+z^2/(2n)-z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}.
\]

The implementation returns zero when no trials are available. Detection recall uses matched ground-truth objects as successes and all ground-truth objects as trials; precision uses accepted correct predictions over all accepted predictions. Events use correctly classified clips over all clips. The formula follows the [NIST description of Wilson intervals](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm). It assumes independent trials; correlated objects/frames can invalidate nominal coverage. Current acceptance checks minimum source groups but does not implement cluster bootstrap intervals or simultaneous multiple-comparison guarantees.

Reviewed replay uses seeded round-robin selection over label-signature buckets. New approved evidence enters only training, and every held-out sample remains unchanged. Hard-example priorities are fixed weights for reviewed missed events (3), false positives (2), true events (1), or unlinked evidence (0). Neither these priorities nor configured acceptance thresholds are mathematically optimal. The [reviewed-learning guide](reviewed-learning.md) records operational boundaries.

## What is established

Tests establish specific geometry, gradient, split-isolation, API and lifecycle properties. Training establishes measured performance on the recorded splits. No theorem or current experiment establishes that this system supports every dataset, always calibrates correctly, is universally better than YOLO, or guarantees perfect surveillance accuracy.
