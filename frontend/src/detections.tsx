import { useEffect, useState } from "react";

type Detection = {
  label: string;
  score: number;
  track_id?: number;
  box?: number[];
};

type Row = Record<string, any>;

export function entityName(label: string) {
  return label === "synthetic_entity"
    ? "Synthetic test entity"
    : label.replaceAll("_", " ");
}

export function detectionSummary(detections: Detection[] = []) {
  if (!detections.length) return "No objects detected in latest inference";
  const counts = new Map<string, number>();
  detections.forEach((item) =>
    counts.set(item.label, (counts.get(item.label) || 0) + 1),
  );
  return [...counts]
    .map(([label, count]) => `${entityName(label)} × ${count}`)
    .join(" · ");
}

export function EventDetails({
  evidence,
}: {
  evidence?: {
    decision?: string;
    reason?: string;
    classes?: string[];
    probabilities?: number[];
    warning?: string;
  };
}) {
  if (!evidence) return null;
  const scores = evidence.probabilities || [];
  const index = scores.length ? scores.indexOf(Math.max(...scores)) : -1;
  return (
    <section className="detected-panel">
      <h3>Video event assessment</h3>
      {index >= 0 && evidence.classes?.[index] ? (
        <p>
          {entityName(evidence.classes[index])} ·{" "}
          {(scores[index] * 100).toFixed(1)}% model estimate
        </p>
      ) : (
        <p>{evidence.reason || "No supported event assessment"}</p>
      )}
      {evidence.warning && <small>{evidence.warning}</small>}
    </section>
  );
}

export function DetectionDetails({
  detections = [],
  model = "Unknown model",
  calibration = "uncalibrated",
}: {
  detections?: Detection[];
  model?: string;
  calibration?: string;
}) {
  return (
    <section className="detected-panel">
      <h3>What was detected</h3>
      <p>{detectionSummary(detections)}</p>
      {detections.length > 0 && (
        <div className="detected-list">
          {detections.map((item, index) => (
            <div className="detected-item" key={`${item.track_id}-${index}`}>
              <strong>{entityName(item.label)}</strong>
              <span>Track {item.track_id ?? "unassigned"}</span>
              <span>
                {item.label === "synthetic_entity"
                  ? "Generated fixture match"
                  : `${calibration === "fitted_detection_correctness" ? "Calibrated detection score" : "Raw model score"}: ${Number(item.score).toFixed(3)}`}
              </span>
            </div>
          ))}
        </div>
      )}
      <small>
        Model: {model}.{" "}
        {calibration === "fitted_detection_correctness"
          ? "Calibration estimates box/class correctness in the evaluation domain."
          : "Raw scores are not calibrated probabilities."}{" "}
        An object detection alone does not establish a threat.
      </small>
    </section>
  );
}

export function EvidenceSequence({
  incidentId,
  title,
  request,
}: {
  incidentId: string;
  title: string;
  request: (path: string) => Promise<Row>;
}) {
  const [sequence, setSequence] = useState<Row | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    setSequence(null);
    setError("");
    request(`/incidents/${incidentId}/frames`)
      .then((value) => current && setSequence(value))
      .catch((reason) => current && setError(String(reason)));
    return () => {
      current = false;
    };
  }, [incidentId, request]);
  if (error) return <p className="error">Recorded sequence could not be loaded.</p>;
  if (!sequence) return <p role="status">Loading recorded sequence…</p>;
  return (
    <section className="evidence-sequence">
      <div>
        <h3>Recorded detection sequence</h3>
        <p>{title || "Detected sequence · review required"}</p>
      </div>
      <div className="evidence-strip">
        {(sequence.frames || []).map((frame: Row) => (
          <figure key={frame.index}>
            <img
              src={frame.review_url || frame.url}
              alt={`${title || "Detected sequence"}, frame ${frame.index + 1}`}
              loading="lazy"
            />
            <figcaption>Frame {frame.index + 1}</figcaption>
          </figure>
        ))}
      </div>
      <small>
        Frames and the titled review clip are encrypted at rest and expire with
        the configured evidence-retention policy. Raw frames remain separate for
        reviewed annotation.
      </small>
    </section>
  );
}
