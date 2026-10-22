import React, { useEffect, useState } from "react";
type Row = Record<string, any>;
const number = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3)
    : "Unavailable";
function Curve({ rows }: { rows: Row[] }) {
  const hasAP = rows.some((r) => Number.isFinite(r.validation_map50));
  const field = hasAP ? "validation_map50" : "validation_loss";
  const label = hasAP ? "Validation AP50" : "Validation loss";
  const values = rows
    .map((r) => r[field])
    .filter((v) => typeof v === "number" && Number.isFinite(v));
  if (values.length < 2) return <p>No recorded validation curve.</p>;
  const ceiling = hasAP ? 1 : Math.max(...values, 0.001);
  return (
    <svg
      viewBox="0 0 400 110"
      role="img"
      aria-label={`${label} by recorded epoch`}
      style={{ width: "100%", maxWidth: 600, height: 140 }}
    >
      <line
        x1="0"
        y1="100"
        x2="400"
        y2="100"
        stroke="currentColor"
        opacity=".2"
      />
      <polyline
        fill="none"
        stroke="#78b6ff"
        strokeWidth="2"
        points={values
          .map(
            (v, i) =>
              `${(i * 400) / Math.max(1, values.length - 1)},${100 - (v / ceiling) * 100}`,
          )
          .join(" ")}
      />
      <text x="4" y="12" fill="currentColor" fontSize="10">
        {label}, scale 0–{ceiling.toFixed(2)}
      </text>
    </svg>
  );
}
export function ModelPerformance({
  request,
}: {
  request: (path: string) => Promise<any>;
}) {
  const [data, setData] = useState<Row | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const value = await request("/model-performance");
        if (alive) {
          setData(value);
          setError("");
        }
      } catch (e) {
        if (alive) setError((e as Error).message);
      } finally {
        if (alive) timer = setTimeout(refresh, 15000);
      }
    };
    refresh();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [request]);
  if (!data)
    return (
      <section className="panel">
        <h2>Model performance</h2>
        <p>{error || "Loading verified model evidence…"}</p>
      </section>
    );
  return (
    <div className="performance">
      <section className="panel">
        <h2>Model performance</h2>
        <p>
          Validation comparisons stay within the same task, domain, dataset and
          taxonomy. Recommendations do not activate models. Missing measurements
          remain unavailable.
        </p>
        {error && <p role="alert">{error}</p>}
        <p>
          <strong>{data.models.length}</strong> completed artifacts ·{" "}
          <strong>
            {Object.values(data.runtime.active_models).filter(Boolean).length}
          </strong>{" "}
          active deployments · <strong>{data.partial_runs.length}</strong>{" "}
          partial runs
        </p>
        <p>
          CPU {number(data.resources.cpu_percent)}% · memory{" "}
          {number(data.resources.memory_percent)}% · GPU{" "}
          {data.resources.gpu?.name || "Unavailable"}{" "}
          {number(Number(data.resources.gpu?.utilization))}% · VRAM{" "}
          {data.resources.gpu?.used_mb ?? "Unavailable"} MiB
        </p>
        <p>
          Mean runtime latency: {number(data.runtime.mean_latency_ms)} ms · p95:{" "}
          {number(data.runtime.p95_latency_ms)} ms
        </p>
      </section>
      <section className="panel">
        <h2>Best compatible model by category</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category / domain</th>
                <th>Recommended version</th>
                <th>Validation AP50 / F1</th>
                <th>Precision / recall / F1</th>
                <th>Missed / false objects</th>
                <th>Deployment</th>
              </tr>
            </thead>
            <tbody>
              {data.recommendations.map((r: Row) => (
                <tr key={r.contract + r.category}>
                  <td>
                    {r.category}
                    <small>
                      {r.task} · {r.domain} · dataset{" "}
                      {r.dataset_sha256?.slice(0, 8)}
                    </small>
                  </td>
                  <td>
                    {r.selected.version}
                    <details>
                      <summary>Why selected</summary>
                      <p>{r.reason}</p>
                      <p>
                        ECE {number(r.selected.calibration_ece)}. Steady latency{" "}
                        {number(r.selected.latency_ms)} ms.
                      </p>
                      <p>
                        Fallbacks:{" "}
                        {r.fallbacks.map((f: Row) => f.version).join(", ") ||
                          "None with compatible evidence"}
                      </p>
                    </details>
                  </td>
                  <td>
                    {number(
                      r.task === "detection"
                        ? r.selected.metrics.ap50
                        : r.selected.metrics.f1,
                    )}
                  </td>
                  <td>
                    {number(r.selected.metrics.precision)} /{" "}
                    {number(r.selected.metrics.recall)} /{" "}
                    {number(r.selected.metrics.f1)}
                  </td>
                  <td>
                    {r.selected.metrics.false_negatives ?? "Unavailable"} /{" "}
                    {r.selected.metrics.false_positives ?? "Unavailable"}
                  </td>
                  <td>
                    {r.selected.stage}
                    {data.models.find(
                      (m: Row) => m.version === r.selected.version,
                    )?.active
                      ? " · loaded"
                      : " · inactive"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!data.recommendations.length && (
          <p>
            No fingerprinted validation comparisons yet. Evaluate the catalog
            before selecting routes.
          </p>
        )}
      </section>
      <section className="panel">
        <h2>Unpromoted trained candidates</h2>
        <p>
          These saved training checkpoints improved some validation classes even
          when their overall score did not beat the parent. They need fresh
          calibration and acceptance before routing. These are not test gains.
        </p>
        {(data.unpromoted_candidates || []).map((candidate: Row) => (
          <details key={candidate.version}>
            <summary>{candidate.version} · validation only · inactive</summary>
            <p>{candidate.scope}</p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Validation objects</th>
                    <th>Parent AP50</th>
                    <th>Candidate AP50</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {candidate.class_improvements.map((row: Row) => (
                    <tr key={row.class}>
                      <td>{row.class}</td>
                      <td>{row.support}</td>
                      <td>{number(row.parent_ap50)}</td>
                      <td>{number(row.candidate_ap50)}</td>
                      <td>{number(row.delta)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!candidate.class_improvements.length && (
              <p>
                No class improved against the same-geometry parent at this
                checkpoint.
              </p>
            )}
          </details>
        ))}
        {!data.unpromoted_candidates?.length && (
          <p>No separately evaluated trained candidates recorded.</p>
        )}
      </section>
      <section className="panel">
        <h2>All model versions</h2>
        {data.models.map((m: Row) => (
          <details key={m.version}>
            <summary>
              {m.version} · {m.manifest.task} · {m.stage}
              {m.active ? " · active" : ""}
            </summary>
            <p>
              {m.manifest.domain} · {m.manifest.classes.join(", ")}
            </p>
            <p>
              Dataset license:{" "}
              {m.dataset_license ||
                "Unavailable — inspect source rights before use"}
            </p>
            <p>
              Held-out test{" "}
              {m.manifest.task === "detection" ? "AP50" : "accuracy"}:{" "}
              {number(
                m.manifest.metrics?.map50 ?? m.manifest.metrics?.accuracy,
              )}
              . Validation mAP50–95: {number(m.evaluation?.metrics?.map50_95)}.
              Validation ECE / Brier / NLL:{" "}
              {number(
                m.evaluation?.metrics?.reliability?.ece ??
                  m.evaluation?.metrics?.ece,
              )}{" "}
              /{" "}
              {number(
                m.evaluation?.metrics?.reliability?.brier ??
                  m.evaluation?.metrics?.brier,
              )}{" "}
              /{" "}
              {number(
                m.evaluation?.metrics?.reliability?.nll ??
                  m.evaluation?.metrics?.nll,
              )}
            </p>
            {m.evaluation?.metrics?.confusion_matrix && (
              <pre>{JSON.stringify(m.evaluation.metrics.confusion_matrix)}</pre>
            )}
            <p>
              Acceptance report:{" "}
              {m.manifest.acceptance?.report_id || "None approved"}.{" "}
              {m.manifest.gate_reason}
            </p>
            <Curve rows={m.learning_curve || []} />
            {!!m.learning_curve?.length && (
              <details>
                <summary>Latest recorded epoch: losses and metrics</summary>
                <pre>{JSON.stringify(m.learning_curve.at(-1), null, 2)}</pre>
              </details>
            )}
            <p>
              {m.evaluation
                ? "Validation evidence verified against manifest and weights."
                : "No comparable validation report; excluded from automatic ranking."}
            </p>
          </details>
        ))}
      </section>
      <section className="panel">
        <h2>Training progress</h2>
        {data.generations.map((g: Row) => (
          <details key={g.generation}>
            <summary>
              {g.generation} · {g.state}
            </summary>
            {g.error && g.state !== "running" && <p role="alert">{g.error}</p>}
            {g.jobs?.map((j: Row) => (
              <div key={j.version || j.id}>
                <strong>{j.version || j.id}</strong> · {j.state}
                {j.error && <p role="alert">{j.error}</p>}
                <p>
                  Epoch {j.completed_epochs ?? "Unavailable"} · loss{" "}
                  {number(j.latest_epoch?.train_loss)} · validation AP50{" "}
                  {number(j.latest_epoch?.validation_map50)}
                </p>
                {j.latest_epoch && (
                  <details>
                    <summary>
                      Recorded losses, LR, assignments and validation metrics
                    </summary>
                    <pre>{JSON.stringify(j.latest_epoch, null, 2)}</pre>
                  </details>
                )}
                {j.comparison && (
                  <p>
                    Held-out change: {number(j.comparison.before)} →{" "}
                    {number(j.comparison.after)} (delta{" "}
                    {number(j.comparison.delta)})
                  </p>
                )}
              </div>
            ))}
          </details>
        ))}
      </section>
      <section className="panel">
        <h2>Runtime routing and remaining evidence</h2>
        <pre>{JSON.stringify(data.runtime.category_routing, null, 2)}</pre>
        {data.unavailable_measurements.map((message: string) => (
          <p key={message}>{message}</p>
        ))}
        {data.errors.map((r: Row) => (
          <p role="alert" key={r.version}>
            {r.version}: {r.error}
          </p>
        ))}
      </section>
    </div>
  );
}
