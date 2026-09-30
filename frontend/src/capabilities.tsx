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
};
type Profile = {
  id: string;
  title: string;
  task: string;
  interpretation: string;
  requirements: string[];
  required_classes: string[];
  datasets: string[];
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
    comparison?: {
      metric: string; before: number; after: number; delta: number;
      classes: { class: string; metric: string; before: number | null; after: number | null; delta: number | null }[];
    };
    coverage?: { class: string; labeled_instances: Record<string, number> }[];
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
                {model.synthetic ? "Synthetic" : "Real data"} · Stage:{" "}
                {model.stage} · {model.active ? "Live" : "Inactive"}
              </p>
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
