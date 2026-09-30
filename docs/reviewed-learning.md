# Reviewed learning and independent acceptance

SurveilX supports object detection and single-label video events. An automated candidate-training loop is available after people establish ground truth. It does not convert its own predictions into verified labels or automatically deploy new models.

## Annotate and review

1. Open an incident and choose **Annotate evidence**. Select the evidence observation and frame.
2. Draw normalized boxes or enter coordinates. Use exact class names from the intended base dataset. Include every object in that taxonomy; explicitly mark empty frames when appropriate.
3. For video events, supply the event class and the same persistent track IDs in each selected frame. The selected frame count and entity count must match the base dataset's input contract. The editor uses neutral context; the API accepts an explicit four-value context vector.
4. Record the domain, camera/session source group and training rights. Footage imported elsewhere from that same source must retain the same group.
5. Submit the annotation. A different researcher or administrator inspects the frames and approves or rejects it under **Reviewed learning**. Reviewed labels are immutable.

Reviewed incident feedback can be linked to an annotation. Missed events receive priority 3, false positives 2, true events 1, and unlinked annotations 0. This is an explicit sampling heuristic, not an estimate of risk or correctness. Ambiguous feedback is excluded. Feedback alone does not supply object boxes or event supervision.

## Build later generations

Choose a base dataset and a new version name. The builder selects compatible approved annotations, prioritizes linked hard examples, and adds them only to `train`. A seeded round-robin over label-signature buckets retains up to the replay limit from the previous training set. This encourages label diversity but does not guarantee equal per-class or per-domain counts.

Validation, calibration and test assets, labels and groups are copied unchanged. The new version records the original manifest hash, annotation author/reviewer, evidence hash, rights and consumed annotation/frame identities. Build validation rejects held-out group overlap, incompatible taxonomies/domains/synthetic scope, changed evidence and repeated evidence frames. Versions publish only after successful validation.

Exact hashes and declared source groups cannot detect all near-duplicates or incorrectly declared origins. Correction of already-consumed labels requires a separately curated replacement dataset; the builder refuses to append the same frame again. Replay is bounded: evaluate retention on earlier domains explicitly.

## Automatic candidate training

Administrators can enable a policy with a base dataset, architecture, epoch count, minimum approved annotation count and replay limits. The single-process worker checks every 15 seconds. Once the threshold is met, it builds a new version and runs the existing train/validate/calibrate/test/register pipeline. After completion, that version becomes the base for the next generation.

Custom models start from random initialization in this dashboard workflow; adapted YOLO starts from its configured pretrained base. Generations reuse reviewed data and replay, not automatically the previous weights. CLI continuation remains available separately. Training never fabricates labels. A failed or interrupted generation requires reviewing its logs and saving the policy to retry; it does not repeatedly train without intervention. Disabling the policy prevents new jobs but does not cancel an already-running job.

The worker runs in one server process. It is not a distributed job queue, and approval-count triggering is not a validated statistical drift detector. Existing validation/calibration/test sets remain useful for regression checks, but repeated use can cause selection bias; final acceptance needs new, independently collected data.

## Independent acceptance

Open **Model acceptance**, choose a candidate, an optional comparison baseline and a new dataset. Set criteria before evaluating. Evaluation uses only that dataset's test split, with weights, calibration and operating threshold frozen. It checks test groups and exact asset hashes against every split of the candidate and baseline training manifests and their adaptation ancestors.

The report records sample/group/class support, AP50 or event accuracy, calibration error, nominal 95% Wilson lower bounds, wall time and baseline regression. Minimum support and source-group counts prevent accepting a trivially small evaluation. Wilson bounds assume independent Bernoulli trials; correlated detections or clips violate that assumption. These bounds are diagnostics, not simultaneous guarantees across all classes, cameras or metrics. See the [mathematical foundations](mathematical-foundations.md).

The service binds acceptance to checkpoint and artifact-manifest hashes, including calibration/configuration. A passing report on real data can be explicitly approved by an administrator. That changes eligibility only. Activate a scoped canary and then promote it in **Model registry**. Promotion, production restoration and rollback to production recheck the approved artifact. Synthetic reports can never authorize production, even when all numerical criteria pass.

This gate does not yet measure false alarms per hour, temporal event localization, operator response, near-duplicate similarity or clustered confidence intervals. No current result establishes universal dataset support, perfect accuracy, or superiority to YOLO.

## Verification

`tests/test_adaptation.py` exercises review separation, atomic dataset assembly, frozen held-out data, taxonomy/group guards, detection and event exports, duplicate evidence, and worker lifecycle. `tests/test_acceptance.py` checks data independence, statistical boundaries, lineage, synthetic rejection and artifact binding.

With the server running, `python -m scripts.verify_acceptance` generates a separate synthetic dataset, submits a real frozen-model evaluation, and verifies that production approval is rejected. Its report is saved to `reports/acceptance-workflow.json`. This creates test evaluation/audit records, not human annotations or approvals.
