type Detection = {
  label: string;
  score: number;
  track_id?: number;
  box?: number[];
};

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

export function DetectionDetails({
  detections = [],
  model = "Unknown model",
}: {
  detections?: Detection[];
  model?: string;
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
                  : `Raw model score: ${Number(item.score).toFixed(3)}`}
              </span>
            </div>
          ))}
        </div>
      )}
      <small>
        Model: {model}. Raw scores are not calibrated probabilities or proof of
        a threat.
      </small>
    </section>
  );
}
