import { useState } from "react";

type Report = {
  count_unit: string;
  scope: string;
  counts: Record<string, number>;
  recorded_groups: Record<string, number>;
  empty_samples: Record<string, number>;
  per_class: { label: string; labeled_instances: Record<string, number>; samples_with_label: Record<string, number> }[];
  issues: { label: string; split: string; kind: string; message: string }[];
  synthetic: boolean;
};

export function DatasetReadiness({ datasets, request }: {
  datasets: { name: string }[];
  request: (path: string) => Promise<any>;
}) {
  const [name, setName] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function inspect() {
    setBusy(true);
    setReport(null);
    setError("");
    try { setReport(await request(`/datasets/${encodeURIComponent(name)}/readiness`)); }
    catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }
  return <section className="panel padded" aria-label="Dataset label coverage">
    <h2>Check dataset label coverage</h2>
    <p>Inspect imported or reviewed data for missing classes and sparse evaluation labels. This checks files and labels without starting training.</p>
    <label>Dataset <select value={name} disabled={busy} onChange={event => { setName(event.target.value); setReport(null); setError(""); }}>
      <option value="">Choose dataset</option>
      {datasets.map(row => <option key={row.name} value={row.name}>{row.name}</option>)}
    </select></label>
    <button type="button" disabled={busy || !name} onClick={() => void inspect()}>{busy ? "Checking files and labels…" : "Check coverage"}</button>
    {error && <p role="alert">{error}</p>}
    {report && <>
      <p>{report.synthetic ? "Synthetic data" : "Imported / reviewed data"} · Counts below are {report.count_unit}. File and split checks passed.</p>
      <div className="table-panel"><table>
        <thead><tr><th>Class</th>{["train", "validation", "calibration", "test"].map(split => <th key={split}>{split}</th>)}</tr></thead>
        <tbody>{report.per_class.map(row => <tr key={row.label}><td>{row.label.replaceAll("_", " ")}</td>
          {["train", "validation", "calibration", "test"].map(split => <td key={split}>{row.labeled_instances[split]} in {row.samples_with_label[split]} samples</td>)}
        </tr>)}</tbody>
      </table></div>
      <p>{Object.entries(report.counts).map(([split, count]) => `${split}: ${count} samples, ${report.recorded_groups[split]} recorded groups, ${report.empty_samples[split]} empty`).join(" · ")}</p>
      {report.issues.length > 0 ? <details open><summary>{report.issues.length} coverage issues to review</summary>
        <ul>{report.issues.map((issue, index) => <li key={index}>{issue.message}</li>)}</ul>
      </details> : <p>Each class has at least ten labels per split. Representative footage and model evaluation are still required.</p>}
      <p>{report.scope}</p>
    </>}
  </section>;
}
