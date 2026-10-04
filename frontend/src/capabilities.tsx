import { useEffect, useState } from "react";

type Candidate = {
  id: string;
  version: string;
  domain: string;
  stage: string;
  synthetic: boolean;
  calibrated: boolean;
  active: boolean;
  acceptance_approved: boolean;
  origin: string;
  classes: string[];
  dataset_provenance?: { staged?: boolean; label_scope?: string };
  development_evidence?: {
    metric: string; score: number | null; target: number; aggregate_target_met: boolean | null;
    all_classes_target_met: boolean;
    per_class: { label: string; metric: string; score: number | null; support: number; target_met: boolean | null }[];
    interpretation: string;
  };
};
type Profile = {
  id: string;
  title: string;
  task: string;
  interpretation: string;
  requirements: string[];
  required_classes: string[];
  datasets: string[];
  dataset_evidence?: { name: string; provenance: {
    label_scope?: string; staged?: boolean; single_room?: boolean;
    limitations?: string; temporal_annotations_reviewed?: boolean;
  }; license?: string }[];
  models: Candidate[];
  production_validated: boolean;
  implemented_rules: string[];
};
type Coverage = {
  profiles: Profile[];
  unreadable_datasets: string[];
  scope: string;
};
type Generation = {
  generation: string;
  state: string;
  config?: { epochs?: number };
  error?: string;
  jobs: {
    architecture: string; version: string; state: string; completed_epochs: number; epochs?: number;
    error?: string;
    target?: { minimum: number; observed?: number; met?: boolean };
    comparison?: {
      metric: string; before: number; after: number; delta: number;
      classes: { class: string; metric: string; before: number | null; after: number | null; delta: number | null }[];
    };
    coverage?: { class: string; labeled_instances: Record<string, number> }[];
    profiles?: {
      tier: string; image_size: number; validation_map50: number; latency_ms: number;
      test_metrics?: { map50: number }; test_is_selection_criterion?: boolean;
    }[];
  }[];
};

export function CapabilitiesPanel({
  request,
}: {
  request: (path: string) => Promise<any>;
}) {
  const [data, setData] = useState<Coverage | null>(null);
  const [error, setError] = useState("");
  const [generations, setGenerations] = useState<Generation[]>([]);
  useEffect(() => {
    let mounted = true;
    let loading = false;
    async function refresh() {
      if (loading) return;
      loading = true;
      try {
        const [value, runs] = await Promise.all([
          request("/capabilities"),
          request("/generations"),
        ]);
        if (mounted) {
          setData(value);
          setGenerations(runs);
          setError("");
        }
      } catch (reason) {
        if (mounted) setError((reason as Error).message);
      } finally {
        loading = false;
      }
    }
    void refresh();
    const timer = setInterval(refresh, 10000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [request]);
  if (error)
    return (
      <div role="alert" className="panel padded">
        {error}
      </div>
    );
  if (!data) return <p role="status">Loading capability evidence…</p>;
  return (
    <>
      <div className="panel padded">
        <h2>Multi-domain capability evidence</h2>
        <p>
          {data.scope}. Each task below separates imported data, model
          candidates, calibration, acceptance and live activation.
        </p>
        <p>
          Downloaded archives appear as datasets only after import. Site
          policies require configured geometry and temporal evidence in addition
          to object detection.
        </p>
        {data.unreadable_datasets.length > 0 && (
          <p role="alert">
            Unreadable dataset records: {data.unreadable_datasets.join(", ")}
          </p>
        )}
      </div>
      {generations.length > 0 && (
        <section className="panel padded" aria-label="Training generations">
          <h2>Training generations</h2>
          <p>
            Progress refreshes every 10 seconds. Completed candidates require
            separate acceptance before production.
          </p>
          {generations.map((run) => (
            <div key={run.generation}>
              <h3>
                {run.generation} · {run.state.replaceAll("_", " ")}
              </h3>
              {run.jobs.map((job) => (
                <div key={job.version || job.architecture}>
                <p>
                  {job.version || job.architecture}: {job.state.replaceAll("_", " ")}
                  {job.completed_epochs > 0
                    ? ` · ${job.completed_epochs}/${job.epochs ?? run.config?.epochs ?? "?"} epochs recorded`
                    : ""}
                </p>
                {job.target && <p>Target development score: at least {job.target.minimum.toFixed(2)} · {job.target.met == null ? "awaiting final evaluation" : job.target.met ? "aggregate target reached; inspect individual classes" : "below target"}.</p>}
                {job.comparison && <>
                  <p>Development {job.comparison.metric === "map50" ? "AP50" : "accuracy"}: {job.comparison.before.toFixed(4)} → {job.comparison.after.toFixed(4)} ({job.comparison.delta >= 0 ? "+" : ""}{job.comparison.delta.toFixed(4)}). Independent acceptance pending.</p>
                  <details>
                    <summary>Class results and label coverage</summary>
                    <div className="table-panel"><table>
                      <thead><tr><th>Detection / event</th><th>Metric</th><th>Parent</th><th>Candidate</th><th>Change</th><th>Train labels</th><th>Test labels</th></tr></thead>
                      <tbody>{job.comparison.classes.map((item) => {
                        const counts = job.coverage?.find((row) => row.class === item.class)?.labeled_instances;
                        return <tr key={item.class}>
                          <td>{item.class.replaceAll("_", " ")}</td><td>{item.metric}</td>
                          <td>{item.before?.toFixed(4) ?? "Unavailable"}</td>
                          <td>{item.after?.toFixed(4) ?? "Not evaluated"}</td>
                          <td>{item.delta == null ? "—" : `${item.delta >= 0 ? "+" : ""}${item.delta.toFixed(4)}`}</td>
                          <td>{counts?.train ?? "—"}</td><td>{counts?.test ?? "—"}</td>
                        </tr>;
                      })}</tbody>
                    </table></div>
                    <p>Classes without test labels cannot establish detection accuracy. Checkpoints are chosen using validation; these test results do not control selection.</p>
                  </details>
                </>}
                {job.profiles && job.profiles.length > 0 && <details>
                  <summary>Validated power profiles</summary>
                  <div className="table-panel"><table>
                    <thead><tr><th>Power tier</th><th>Input size</th><th>Validation AP50</th><th>Latency</th><th>Test AP50</th></tr></thead>
                    <tbody>{job.profiles.map((profile) => <tr key={profile.tier}>
                      <td>{profile.tier}</td>
                      <td>{profile.image_size} px</td>
                      <td>{profile.validation_map50.toFixed(4)}</td>
                      <td>{profile.latency_ms.toFixed(2)} ms</td>
                      <td>{profile.test_metrics?.map50.toFixed(4) ?? "Pending"}</td>
                    </tr>)}</tbody>
                  </table></div>
                  <p>Calibration is fitted separately for each input size. Validation AP50 and measured latency choose the tier; test AP50 is reported afterward.</p>
                </details>}
                {job.error && <p role="alert">{job.error}</p>}
                </div>
              ))}
              {run.error && <p role="alert">{run.error}</p>}
            </div>
          ))}
        </section>
      )}
      {data.profiles.map((profile) => (
        <section
          className="panel padded"
          key={profile.id}
          aria-labelledby={`capability-${profile.id}`}
        >
          <h2 id={`capability-${profile.id}`}>{profile.title}</h2>
          <p>{profile.interpretation}</p>
          <p>
            <strong>
              {profile.task === "policy"
                ? `Available rule primitives: ${profile.implemented_rules.join(", ") || "none"}; full task validation pending`
                : profile.models.length
                  ? `${profile.models.length} candidate(s) available`
                  : "No task-specific candidate"}
            </strong>{" "}
            ·{" "}
            {profile.production_validated
              ? "Production acceptance verified for the listed model domain"
              : "Real-domain production acceptance pending"}
          </p>
          <p>Required labels: {profile.required_classes.join(", ")}</p>
          {profile.dataset_evidence?.filter((item) => item.provenance.label_scope === "inherited_video_label").map((item) => (
            <p key={item.name}><strong>{item.name}: weak video labels.</strong>{" "}
              Crop intervals have not been individually reviewed.
              {item.provenance.staged ? " Actions are staged." : ""}
              {item.provenance.single_room ? " All recordings share one room." : ""}
              {item.license ? ` Rights: ${item.license}.` : ""}
            </p>
          ))}
          <p>
            Imported dataset declarations:{" "}
            {profile.datasets.length ? profile.datasets.join(", ") : "None"}
          </p>
          {profile.models.map((model) => (
            <div key={model.id}>
              <h3>{model.version}</h3>
              <p>Origin: {model.origin === "external_pretrained" ? "Inherited pretrained model; no local training" : "Locally trained candidate"}</p>
              <details><summary>Model labels ({model.classes.length})</summary><p>{model.classes.join(", ")}</p></details>
              <p>
                Domain: {model.domain} ·{" "}
                {model.synthetic ? "Synthetic" : model.dataset_provenance?.staged ? "Staged recordings" : "Recorded/pretrained data"} · Stage:{" "}
                {model.stage} · {model.active ? "Live" : "Inactive"}
              </p>
              {model.development_evidence && <details>
                <summary>Development {model.development_evidence.metric}: {model.development_evidence.score?.toFixed(4) ?? "Not evaluated"}
                  {" · "}{model.development_evidence.all_classes_target_met ? "All class point scores reach 0.50" : "All-class target not established"}
                </summary>
                <div className="table-panel"><table>
                  <thead><tr><th>Class</th><th>Metric</th><th>Test score</th><th>Test labels</th><th>0.50 floor</th></tr></thead>
                  <tbody>{model.development_evidence.per_class.map((row) => <tr key={row.label}>
                    <td>{row.label.replaceAll("_", " ")}</td><td>{row.metric}</td>
                    <td>{row.score?.toFixed(4) ?? "Unavailable"}</td><td>{row.support}</td>
                    <td>{row.target_met == null ? "Not evaluated" : row.target_met ? "Reached" : "Below"}</td>
                  </tr>)}</tbody>
                </table></div>
                <p>{model.development_evidence.interpretation}</p>
                {model.dataset_provenance?.label_scope === "inherited_video_label" && <p>Training used inherited video labels; individual crop intervals require review.</p>}
              </details>}
              <p>
                Calibration: {model.calibrated ? "Fitted" : "Pending"} ·
                Acceptance:{" "}
                {model.acceptance_approved
                  ? "Approved artifact verified"
                  : "Pending"}
              </p>
            </div>
          ))}
          <details>
            <summary>Data and evaluation requirements</summary>
            <ul>
              {profile.requirements.map((value) => (
                <li key={value}>{value}</li>
              ))}
            </ul>
          </details>
        </section>
      ))}
    </>
  );
}
