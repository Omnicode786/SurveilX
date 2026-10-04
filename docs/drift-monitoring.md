# Drift monitoring

SurveilX monitors changes in observed camera and model behavior without converting predictions into training labels. Each stream is scoped by camera, expert slot and model version so deployments with different semantics are never pooled.

The monitor compares two non-overlapping windows: 60 reference decisions followed by 30 recent decisions. It measures motion, normalized brightness, inference latency, detection count and mean calibrated detection confidence with fixed bins. A stream stays in `collecting` until both windows contain valid observations.

For discrete distributions \(P\) and \(Q\), the effect size is the base-2 Jensen-Shannon divergence

\[
JSD(P,Q)=\frac{1}{2}D_{KL}(P\|M)+\frac{1}{2}D_{KL}(Q\|M),\quad M=\frac{P+Q}{2}.
\]

Jeffreys smoothing adds one half-count per bin before normalization, keeping the divergence finite. A feature signals drift only when both conditions hold:

1. Jensen-Shannon divergence is at least 0.10.
2. A 199-round two-sample permutation test has \(p \leq 0.05/5=0.01\), a Bonferroni correction for the five monitored features.

This is an operational review trigger, not proof of model error or a deployment-quality threshold. A trigger requests human-reviewed evidence. Candidate training still requires compatible, independently approved annotations; the untouched validation, calibration and test splits remain unchanged.

The runtime evaluates drift once per minute and stores the latest bounded result as `drift-status`. Researchers can inspect it through `/api/adaptation/drift` or the Learning page. Raw frames are not duplicated by the monitor; incident evidence remains governed by the existing encrypted evidence and retention controls.
