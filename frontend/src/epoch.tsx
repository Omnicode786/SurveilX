export type Epoch = {
  epoch?: number;
  train_loss?: number;
  validation_loss?: number;
  validation_map50?: number;
  learning_rate?: number;
  metric_source?: string;
  elapsed_seconds?: number;
  training?: {
    losses?: Record<string, number | null>;
    positive_assignments?: number;
    positives_per_level?: number[];
    unassigned_objects?: number;
  };
  validation_metrics?: {
    map50?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    per_class?: { class_id: number; class_name?: string; ap50: number | null }[];
  };
  validation?: { losses?: Record<string, number | null> };
  selection?: { best_epoch?: number; selected?: string; stale_epochs?: number; stopping_reason?: string | null };
};

const value = (number: number | null | undefined) =>
  number != null && Number.isFinite(number) ? number.toPrecision(4) : "Not recorded";

export function EpochDetails({ epoch }: { epoch?: Epoch | null }) {
  if (!epoch) return null;
  const metrics = epoch.validation_metrics;
  const losses: Record<string, number | null | undefined> = { loss: epoch.train_loss, ...epoch.training?.losses };
  const validationLosses: Record<string, number | null | undefined> = { loss: epoch.validation_loss, ...epoch.validation?.losses };
  return <details>
    <summary>Latest recorded epoch {epoch.epoch ?? ""}</summary>
    <p>Validation AP50: {value(epoch.validation_map50)} · Precision: {value(metrics?.precision)} · Recall: {value(metrics?.recall)} · F1: {value(metrics?.f1)}</p>
    <p>Learning rate: {value(epoch.learning_rate)} · Elapsed: {value(epoch.elapsed_seconds)} seconds</p>
    <div className="table-panel"><table>
      <thead><tr><th>Objective</th><th>Training loss</th><th>Validation loss</th></tr></thead>
      <tbody>{Object.entries(epoch.metric_source ? { box_regression: "Box regression", classification: "Classification", distribution_focal: "Distribution focal" } : { loss: "Total", objectness: "Objectness", classification: "Classification", box_giou: "GIoU", box_l1: "Smooth L1" }).map(([key, label]) =>
        <tr key={key}><td>{label}</td><td>{value(losses[key])}</td><td>{value(validationLosses[key])}</td></tr>)}</tbody>
    </table></div>
    {epoch.training?.positive_assignments != null && <p>Positive assignments: {epoch.training.positive_assignments} · Per pyramid level: {epoch.training.positives_per_level?.join(" / ") ?? "Not recorded"} · Objects without supervision: {epoch.training.unassigned_objects ?? "Not recorded"}</p>}
    {epoch.selection && <p>Best epoch: {epoch.selection.best_epoch ?? "Not recorded"} · Selected: {epoch.selection.selected ?? "Not recorded"} · Stale epochs: {epoch.selection.stale_epochs ?? "Not recorded"} · Stop: {epoch.selection.stopping_reason?.replaceAll("_", " ") ?? "Not recorded"}</p>}
    {metrics?.per_class && <div className="table-panel"><table>
      <thead><tr><th>Class</th><th>Validation AP50</th></tr></thead>
      <tbody>{metrics.per_class.map(row => <tr key={row.class_id}><td>{row.class_name ?? row.class_id}</td><td>{value(row.ap50)}</td></tr>)}</tbody>
    </table></div>}
    <p>Older runs may lack component metrics. These fields describe recorded epochs; final evaluation uses the selected checkpoint.</p>
    <p>Training and validation may use different loss recipes. {epoch.metric_source && "YOLO F1 is calculated from its recorded validation precision and recall."}</p>
  </details>;
}
