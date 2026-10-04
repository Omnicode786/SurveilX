import React, { useEffect, useState } from "react";
import "./learning.css";

type Row = Record<string, any>;
type Request = (path: string, method?: string, body?: unknown) => Promise<any>;
type Box = { box: number[]; label: string; track_id: number | null };
type Frame = { index: number; boxes: Box[] };

function EvidenceImage({
  url,
  boxes,
  onBox,
}: {
  url: string;
  boxes: Box[];
  onBox?: (box: number[]) => void;
}) {
  const [start, setStart] = useState<number[] | null>(null);
  const [end, setEnd] = useState<number[] | null>(null);
  function point(e: React.PointerEvent<HTMLDivElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    ];
  }
  const drawn =
    start && end
      ? [
          Math.min(start[0], end[0]),
          Math.min(start[1], end[1]),
          Math.max(start[0], end[0]),
          Math.max(start[1], end[1]),
        ]
      : null;
  return (
    <div
      className={"annotation-canvas " + (onBox ? "editable" : "")}
      onPointerDown={(e) => {
        if (onBox) {
          e.currentTarget.setPointerCapture(e.pointerId);
          setStart(point(e));
          setEnd(point(e));
        }
      }}
      onPointerMove={(e) => {
        if (start) setEnd(point(e));
      }}
      onPointerCancel={() => {
        setStart(null);
        setEnd(null);
      }}
      onPointerUp={(e) => {
        if (start && onBox) {
          const p = point(e);
          const b = [
            Math.min(start[0], p[0]),
            Math.min(start[1], p[1]),
            Math.max(start[0], p[0]),
            Math.max(start[1], p[1]),
          ];
          if (b[2] - b[0] > 0.005 && b[3] - b[1] > 0.005) onBox(b);
        }
        setStart(null);
        setEnd(null);
      }}
    >
      <img src={url} alt="Evidence frame for annotation" draggable={false} />
      <svg
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {[
          ...boxes,
          ...(drawn ? [{ box: drawn, label: "New box", track_id: null }] : []),
        ].map((b, i) => (
          <g key={i}>
            <rect
              x={b.box[0] * 1000}
              y={b.box[1] * 1000}
              width={(b.box[2] - b.box[0]) * 1000}
              height={(b.box[3] - b.box[1]) * 1000}
            />
            <text x={b.box[0] * 1000 + 5} y={Math.max(24, b.box[1] * 1000 - 8)}>
              {b.label}
              {b.track_id !== null ? ` #${b.track_id}` : ""}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function AnnotationEditor({
  incident,
  request,
  close,
  done,
}: {
  incident: Row;
  request: Request;
  close: () => void;
  done: () => void;
}) {
  const [evidence, setEvidence] = useState<Row>({ frames: [] });
  const [feedback, setFeedback] = useState<Row[]>([]);
  const [observation, setObservation] = useState(0),
    [index, setIndex] = useState(0);
  const [frames, setFrames] = useState<Frame[]>([]),
    [task, setTask] = useState("detection");
  const [label, setLabel] = useState("person"),
    [track, setTrack] = useState(0);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true;
    setFrames([]);
    setIndex(0);
    setEvidence({ frames: [] });
    setError("");
    request("/feedback")
      .then((r) => {
        if (live)
          setFeedback(
            r.filter(
              (f: Row) =>
                f.incident_id === incident.id &&
                f.reviewed &&
                f.label !== "ambiguous",
            ),
          );
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    request(`/incidents/${incident.id}/frames?observation=${observation}`)
      .then((r) => {
        if (live) setEvidence(r);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [incident.id, observation, request]);
  const current = frames.find((f) => f.index === index);
  function setBoxes(boxes: Box[]) {
    setFrames((old) =>
      [...old.filter((f) => f.index !== index), { index, boxes }].sort(
        (a, b) => a.index - b.index,
      ),
    );
  }
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const values = Object.fromEntries(new FormData(e.currentTarget));
    try {
      if (!frames.length)
        throw new Error(
          "Annotate at least one frame or explicitly mark it empty.",
        );
      await request("/adaptation/annotations", "POST", {
        incident_id: incident.id,
        feedback_id: values.feedback_id || null,
        observation,
        task,
        domain: values.domain,
        source_group: values.source_group,
        rights: values.rights,
        complete_annotation: values.complete === "on",
        event_label: task === "event" ? values.event_label : null,
        frames,
      });
      done();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <form
        className="dialog wide learning-form"
        role="dialog"
        aria-modal="true"
        aria-label="Annotate evidence"
        onSubmit={submit}
      >
        <button
          type="button"
          className="close icon-btn"
          aria-label="Close annotation"
          onClick={close}
        >
          ×
        </button>
        <p className="eyebrow">REVIEWED TRAINING DATA</p>
        <h2>Annotate evidence</h2>
        <p>
          Draw every object in the target taxonomy. Video events also require
          consistent track IDs across the selected frames. A second person
          reviews the labels before training.
        </p>
        {error && (
          <p role="alert" className="learning-error">
            {error}
          </p>
        )}
        <div className="form-row">
          <label>
            Observation
            <select
              value={observation}
              onChange={(e) => setObservation(Number(e.target.value))}
            >
              {Array.from(
                {
                  length:
                    1 + (incident.details?.additional_evidence?.length || 0),
                },
                (_, i) => (
                  <option key={i} value={i}>
                    {i + 1}
                  </option>
                ),
              )}
            </select>
          </label>
          <label>
            Task
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="detection">Object detection</option>
              <option value="event">Video event</option>
            </select>
          </label>
          <label>
            Frame
            <select
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
            >
              {evidence.frames.map((f: Row) => (
                <option key={f.index} value={f.index}>
                  {f.index + 1}
                  {frames.some((x) => x.index === f.index) ? " · included" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>
        {evidence.frames[index] && (
          <EvidenceImage
            url={evidence.frames[index].url}
            boxes={current?.boxes || []}
            onBox={(box) => {
              if (label.trim())
                setBoxes([
                  ...(current?.boxes || []),
                  {
                    box,
                    label: label.trim(),
                    track_id: task === "event" ? track : null,
                  },
                ]);
            }}
          />
        )}
        <div className="form-row">
          <label>
            Object class
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              required
              maxLength={100}
            />
          </label>
          {task === "event" && (
            <label>
              Track ID
              <input
                type="number"
                min={0}
                value={track}
                onChange={(e) => setTrack(Number(e.target.value))}
              />
            </label>
          )}
        </div>
        <p className="muted">
          Drag on the image to draw a box, or add normalized coordinates below.
          Class names must match your base dataset.
        </p>
        <CoordinateBox
          onAdd={(box) =>
            setBoxes([
              ...(current?.boxes || []),
              {
                box,
                label: label.trim(),
                track_id: task === "event" ? track : null,
              },
            ])
          }
        />
        <div className="annotation-boxes">
          {current?.boxes.map((b, i) => (
            <div key={i}>
              <span>
                {b.label}
                {b.track_id !== null ? ` #${b.track_id}` : ""} ·{" "}
                {b.box.map((x) => x.toFixed(3)).join(", ")}
              </span>
              <button
                type="button"
                onClick={() =>
                  setBoxes(current.boxes.filter((_, j) => i !== j))
                }
              >
                Remove box {i + 1}
              </button>
            </div>
          ))}
        </div>
        <div className="action-row">
          <button type="button" onClick={() => setBoxes([])}>
            Mark frame as empty
          </button>
          <button
            type="button"
            onClick={() =>
              setFrames((old) => old.filter((f) => f.index !== index))
            }
          >
            Exclude frame
          </button>
          <span>
            {frames.length} frames included ·{" "}
            {evidence.synthetic ? "synthetic evidence" : "recorded evidence"}
          </span>
        </div>
        <div className="form-row">
          <label>
            Dataset domain
            <input
              name="domain"
              placeholder="Exact domain from base dataset"
              required
              maxLength={100}
            />
          </label>
          <label>
            Source group
            <input
              name="source_group"
              defaultValue={incident.camera_id}
              required
              maxLength={200}
            />
          </label>
        </div>
        <p className="muted">
          Keep all footage from the same camera/session in one group, including
          footage imported separately.
        </p>
        {task === "event" && (
          <label>
            Event class
            <input
              name="event_label"
              required
              placeholder="Exact event class from base dataset"
            />
          </label>
        )}
        <label>
          Linked reviewed feedback
          <select name="feedback_id">
            <option value="">No linked feedback</option>
            {feedback.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label.replaceAll("_", " ")} · {f.notes || f.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Rights to use this evidence for training
          <input
            name="rights"
            minLength={8}
            maxLength={1000}
            required
            placeholder="Record the owner, permission or license"
          />
        </label>
        <label className="check">
          <input name="complete" type="checkbox" required /> All objects in the
          intended taxonomy are labeled in every included frame.
        </label>
        <button className="primary" disabled={busy || !evidence.frames.length}>
          {busy ? "Submitting…" : "Submit for independent review"}
        </button>
      </form>
    </div>
  );
}

function CoordinateBox({ onAdd }: { onAdd: (box: number[]) => void }) {
  const [values, setValues] = useState([0.1, 0.1, 0.8, 0.8]);
  return (
    <div className="coordinate-fields">
      {["Left", "Top", "Right", "Bottom"].map((label, i) => (
        <label key={label}>
          {label}
          <input
            aria-label={`Box ${label.toLowerCase()}`}
            type="number"
            min={0}
            max={1}
            step={0.001}
            value={values[i]}
            onChange={(e) =>
              setValues((v) =>
                v.map((x, j) => (i === j ? Number(e.target.value) : x)),
              )
            }
          />
        </label>
      ))}
      <button
        type="button"
        disabled={values[2] <= values[0] || values[3] <= values[1]}
        onClick={() => onAdd([...values])}
      >
        Add box
      </button>
    </div>
  );
}

function AnnotationReview({
  record,
  user,
  request,
  close,
  done,
}: {
  record: Row;
  user: Row;
  request: Request;
  close: () => void;
  done: () => void;
}) {
  const p = record.payload,
    [index, setIndex] = useState(0),
    [notes, setNotes] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const frame = p.frames[index];
  async function review(approve: boolean) {
    setBusy(true);
    try {
      await request(`/adaptation/annotations/${record.id}/review`, "POST", {
        approve,
        notes,
      });
      done();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="dialog wide learning-form"
        role="dialog"
        aria-modal="true"
        aria-label="Review annotation"
      >
        <button
          className="close icon-btn"
          aria-label="Close review"
          onClick={close}
        >
          ×
        </button>
        <p className="eyebrow">INDEPENDENT REVIEW</p>
        <h2>{p.task === "event" ? p.event_label : "Object annotations"}</h2>
        <p>
          {p.domain} · {p.synthetic ? "Synthetic" : "Recorded"} · {p.state}
        </p>
        <label>
          Annotated frame
          <select
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
          >
            {p.frames.map((f: Frame, i: number) => (
              <option key={i} value={i}>
                Frame {f.index + 1} · {f.boxes.length} objects
              </option>
            ))}
          </select>
        </label>
        <EvidenceImage
          url={`/api/incidents/${p.incident_id}/frames/${frame.index}?observation=${p.observation}`}
          boxes={frame.boxes}
        />
        <p>
          {frame.boxes
            .map(
              (b: Box) =>
                b.label + (b.track_id !== null ? ` #${b.track_id}` : ""),
            )
            .join(", ") || "Explicitly labeled empty frame"}
        </p>
        <p>Source group: {p.source_group}</p>
        <p>Rights: {p.rights}</p>
        <p>{p.review_notes}</p>
        {error && (
          <p role="alert" className="learning-error">
            {error}
          </p>
        )}
        {p.state === "pending" &&
        user.id !== p.author &&
        ["admin", "researcher"].includes(user.role) ? (
          <>
            <label>
              Review notes
              <textarea
                value={notes}
                maxLength={4000}
                onChange={(e) => setNotes(e.target.value)}
              />
            </label>
            <div className="action-row">
              <button disabled={busy} onClick={() => review(false)}>
                Reject labels
              </button>
              <button
                className="primary"
                disabled={busy}
                onClick={() => review(true)}
              >
                Approve labels
              </button>
            </div>
          </>
        ) : (
          <p className="muted">
            {p.state === "pending"
              ? "A different researcher or administrator must review these annotations."
              : "Reviewed annotations are immutable."}
          </p>
        )}
      </section>
    </div>
  );
}

export function LearningPanel({
  request,
  user,
}: {
  request: Request;
  user: Row;
}) {
  const [annotations, setAnnotations] = useState<Row[]>([]),
    [feedback, setFeedback] = useState<Row[]>([]),
    [datasets, setDatasets] = useState<Row[]>([]),
    [policy, setPolicy] = useState<Row>({}),
    [drift, setDrift] = useState<Row>({ state: "collecting", streams: [] });
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [selected, setSelected] = useState<Row | null>(null),
    [busy, setBusy] = useState(false);
  const advanced = ["admin", "researcher"].includes(user.role);
  async function refresh() {
    const [a, d, p, f, shift] = await Promise.all([
      request("/adaptation/annotations"),
      request("/datasets"),
      advanced ? request("/adaptation/policy") : Promise.resolve({}),
      request("/feedback"),
      advanced ? request("/adaptation/drift") : Promise.resolve({ state: "collecting", streams: [] }),
    ]);
    setAnnotations(a);
    setDatasets(d);
    setPolicy(p);
    setFeedback(f);
    setDrift(shift);
  }
  useEffect(() => {
    if (user.role === "viewer") return;
    const update = () => refresh().catch((e) => setError(e.message));
    update();
    const timer = setInterval(update, 5000);
    return () => clearInterval(timer);
  }, [request, user.role]);
  async function submit(e: React.FormEvent<HTMLFormElement>, mode: string) {
    e.preventDefault();
    const v = Object.fromEntries(new FormData(e.currentTarget));
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const body =
        mode === "datasets"
          ? {
              name: v.name,
              base_dataset: v.base_dataset,
              replay_limit: Number(v.replay_limit),
              max_new: Number(v.max_new),
            }
          : {
              enabled: v.enabled === "on",
              base_dataset: v.base_dataset,
              architecture: v.architecture,
              epochs: Number(v.epochs),
              minimum_approved: Number(v.minimum_approved),
              max_new: Number(v.max_new),
              replay_limit: Number(v.replay_limit),
            };
      const r = await request(`/adaptation/${mode}`, "POST", body);
      setNotice(
        mode === "datasets"
          ? `Created ${r.name} from ${r.annotation_ids.length} approved annotations.`
          : "Automatic training policy saved.",
      );
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  if (user.role === "viewer")
    return (
      <div className="panel padded">
        An operator, researcher or administrator account is required to manage
        training annotations.
      </div>
    );
  return (
    <div className="learning-workspace">
      {error && (
        <p role="alert" className="learning-error">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="learning-notice">
          {notice}
        </p>
      )}
      <div className="learning-steps">
        <span>01 · Annotate evidence</span>
        <span>02 · Independent review</span>
        <span>03 · Train & calibrate</span>
        <span>04 · Evaluate & approve</span>
      </div>
      {advanced && (
        <section className="panel padded">
          <div className="section-heading">
            <div>
              <p className="eyebrow">DISTRIBUTION MONITORING</p>
              <h2>Camera and model drift</h2>
            </div>
            <span className="tag">{String(drift.state || "collecting").replaceAll("_", " ")}</span>
          </div>
          <p>{drift.interpretation || "Collecting non-overlapping reference and recent observation windows."}</p>
          {drift.streams?.length > 0 && <div className="table-panel"><table>
            <thead><tr><th>Camera</th><th>Model</th><th>State</th><th>Evidence</th></tr></thead>
            <tbody>{drift.streams.map((stream: Row) => <tr key={`${stream.camera_id}-${stream.expert_slot}-${stream.model}`}>
              <td>{stream.camera_id}</td><td>{stream.model}</td>
              <td>{String(stream.state).replaceAll("_", " ")}</td>
              <td>{stream.shifted_features?.length ? `Shift: ${stream.shifted_features.join(", ")}` : `${stream.observations}/${stream.required || stream.observations} observations`}</td>
            </tr>)}</tbody>
          </table></div>}
          <p>A signal requires a material Jensen–Shannon shift and a Bonferroni-corrected permutation test. It requests reviewed evidence and never creates training labels automatically.</p>
        </section>
      )}
      <section className="panel padded">
        <div className="section-heading">
          <div>
            <p className="eyebrow">GROUND TRUTH</p>
            <h2>Reviewed annotations</h2>
          </div>
          <span className="tag">
            {annotations.filter((a) => a.payload.state === "approved").length}{" "}
            approved
          </span>
        </div>
        <p>
          Open an incident and choose “Annotate evidence” to add object boxes or
          a video-event label. Predictions alone are never accepted as ground
          truth.
        </p>
        {annotations.length ? (
          <div className="learning-records">
            {annotations.map((a) => (
              <button
                className="learning-record"
                key={a.id}
                onClick={() => setSelected(a)}
              >
                <span>
                  <strong>{a.payload.event_label || "Object detection"}</strong>
                  <small>
                    {a.payload.domain} · {a.payload.frames.length} frames ·{" "}
                    {a.payload.synthetic ? "synthetic" : "recorded"}
                  </small>
                </span>
                <span className="tag">{a.payload.state}</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="learning-empty">
            No annotations yet. Your reviewed evidence will appear here.
          </div>
        )}
      </section>
      {advanced && feedback.some((f) => !f.reviewed) && (
        <section className="panel padded">
          <h2>Feedback awaiting review</h2>
          <p>
            Reviewed missed events and false positives can prioritize linked
            annotations in the next training generation.
          </p>
          <div className="learning-records">
            {feedback
              .filter((f) => !f.reviewed)
              .map((f) => (
                <div className="learning-record" key={f.id}>
                  <span>
                    <strong>{f.label.replaceAll("_", " ")}</strong>
                    <small>
                      {f.notes || "No notes"} · Incident{" "}
                      {f.incident_id.slice(0, 8)}
                    </small>
                  </span>
                  <button
                    disabled={busy || f.actor === user.id}
                    onClick={async () => {
                      setBusy(true);
                      try {
                        await request(`/feedback/${f.id}/review`, "POST");
                        await refresh();
                      } catch (e: any) {
                        setError(e.message);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Review feedback
                  </button>
                </div>
              ))}
          </div>
        </section>
      )}
      {advanced && (
        <div className="learning-grid">
          <form
            className="panel padded learning-form"
            onSubmit={(e) => submit(e, "datasets")}
          >
            <p className="eyebrow">DATASET GENERATIONS</p>
            <h2>Build a reviewed dataset</h2>
            <p>
              Add compatible approved labels to training and retain a bounded,
              label-diverse sample of earlier training data. Existing
              validation, calibration and test samples stay fixed.
            </p>
            <DatasetSelect datasets={datasets} />
            <label>
              New version name
              <input
                name="name"
                required
                pattern="[a-zA-Z0-9_-]{1,64}"
                placeholder="reviewed-generation-01"
              />
            </label>
            <ReplayFields />
            <button className="primary" disabled={busy || !datasets.length}>
              Build dataset version
            </button>
          </form>
          <form
            key={
              policy.configured_by
                ? `${policy.configured_by}-${policy.base_dataset}-${policy.enabled}`
                : "new"
            }
            className="panel padded learning-form"
            onSubmit={(e) => submit(e, "policy")}
          >
            <p className="eyebrow">AUTOMATIC TRAINING</p>
            <h2>Learn from approved data</h2>
            <p>
              When enough new labels pass review, build a version, train a
              candidate and fit calibration. Deployment requires a separate
              acceptance decision.
            </p>
            <p className="tag">
              {(policy.state || "Not configured").replaceAll("_", " ")}
            </p>
            {policy.last_error && (
              <p className="learning-error">{policy.last_error}</p>
            )}
            <fieldset disabled={user.role !== "admin" || busy}>
              <DatasetSelect datasets={datasets} value={policy.base_dataset} />
              <label>
                Architecture
                <select
                  name="architecture"
                  defaultValue={policy.architecture || "auto"}
                >
                  <option value="auto">
                    Custom architecture for dataset task
                  </option>
                  <option value="scratch">Custom model from scratch</option>
                  <option value="yolo_rai">
                    Adapted YOLO · detection only
                  </option>
                </select>
              </label>
              <div className="form-row">
                <label>
                  Training epochs
                  <input
                    type="number"
                    name="epochs"
                    min={1}
                    max={500}
                    defaultValue={policy.epochs || 5}
                    required
                  />
                </label>
                <label>
                  Minimum approved annotations
                  <input
                    type="number"
                    name="minimum_approved"
                    min={1}
                    max={500}
                    defaultValue={policy.minimum_approved || 10}
                    required
                  />
                </label>
              </div>
              <ReplayFields policy={policy} />
              <label className="check">
                <input
                  type="checkbox"
                  name="enabled"
                  defaultChecked={policy.enabled}
                />{" "}
                Enable automatic candidate training
              </label>
              <button className="primary" disabled={!datasets.length}>
                Save training policy
              </button>
            </fieldset>
          </form>
        </div>
      )}
      {selected && (
        <AnnotationReview
          record={selected}
          user={user}
          request={request}
          close={() => setSelected(null)}
          done={() => {
            setSelected(null);
            refresh().catch((e) => setError(e.message));
          }}
        />
      )}
    </div>
  );
}

function DatasetSelect({
  datasets,
  value,
}: {
  datasets: Row[];
  value?: string;
}) {
  return (
    <label>
      Base dataset
      <select name="base_dataset" required defaultValue={value || ""}>
        <option value="" disabled>
          Select a dataset
        </option>
        {datasets.map((d) => (
          <option key={d.name} value={d.name}>
            {d.name} · {d.task || "event"} · {d.domain}
          </option>
        ))}
      </select>
    </label>
  );
}
function ReplayFields({ policy = {} }: { policy?: Row }) {
  return (
    <div className="form-row">
      <label>
        Replay sample limit
        <input
          type="number"
          name="replay_limit"
          min={10}
          max={10000}
          defaultValue={policy.replay_limit || 500}
          required
        />
      </label>
      <label>
        Maximum new annotations
        <input
          type="number"
          name="max_new"
          min={1}
          max={500}
          defaultValue={policy.max_new || 100}
          required
        />
      </label>
    </div>
  );
}

const thresholds = [
  ["min_samples", "Minimum test samples", 100, 10, 1000000, 1],
  ["min_groups", "Minimum source groups", 5, 2, 100000, 1],
  ["min_per_class", "Minimum labels per class", 20, 5, 100000, 1],
  ["min_ap50", "Minimum AP50 · detection", 0.8, 0, 1, 0.01],
  ["min_recall_lower", "Minimum recall lower bound", 0.8, 0, 1, 0.01],
  ["min_precision_lower", "Minimum precision lower bound", 0.8, 0, 1, 0.01],
  [
    "min_accuracy_lower",
    "Minimum accuracy lower bound · events",
    0.8,
    0,
    1,
    0.01,
  ],
  ["max_ece", "Maximum calibration error", 0.08, 0, 1, 0.01],
  ["max_wall_ms", "Maximum wall time per sample · ms", 500, 1, 60000, 1],
  ["max_regression", "Maximum baseline regression", 0.02, 0, 0.2, 0.01],
] as const;

export function AcceptancePanel({
  request,
  user,
}: {
  request: Request;
  user: Row;
}) {
  const [models, setModels] = useState<Row[]>([]),
    [datasets, setDatasets] = useState<Row[]>([]),
    [reports, setReports] = useState<Row[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const permitted = ["admin", "researcher"].includes(user.role);
  async function refresh() {
    const [m, d, r] = await Promise.all([
      request("/models"),
      request("/datasets"),
      request("/acceptance"),
    ]);
    setModels(m);
    setDatasets(d);
    setReports(r);
  }
  useEffect(() => {
    if (!permitted) return;
    const update = () => refresh().catch((e) => setError(e.message));
    update();
    const timer = setInterval(update, 5000);
    return () => clearInterval(timer);
  }, [request, permitted]);
  async function evaluate(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const v = Object.fromEntries(new FormData(e.currentTarget));
    try {
      await request(`/models/${v.model}/evaluate`, "POST", {
        dataset: v.dataset,
        baseline_model_id: v.baseline || null,
        policy: Object.fromEntries(
          thresholds.map(([key]) => [key, Number(v[key])]),
        ),
      });
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function approve(id: string) {
    setBusy(true);
    setError("");
    try {
      await request(`/acceptance/${id}/approve`, "POST");
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  if (!permitted)
    return (
      <div className="panel padded">
        A researcher or administrator account is required to view acceptance
        evaluations.
      </div>
    );
  return (
    <div className="learning-workspace">
      {error && (
        <p role="alert" className="learning-error">
          {error}
        </p>
      )}
      <form className="panel padded learning-form" onSubmit={evaluate}>
        <p className="eyebrow">INDEPENDENT EVALUATION</p>
        <h2>Measure before deployment</h2>
        <p>
          Evaluate frozen weights and calibration on a new dataset’s test split.
          Training lineage, source groups and exact file hashes are checked for
          overlap. Synthetic results cannot qualify a model for production.
        </p>
        <fieldset disabled={user.role !== "admin" || busy}>
          <div className="learning-grid">
            <label>
              Candidate
              <select name="model" required>
                <option value="">Choose model</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · {m.version.slice(0, 24)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Comparison baseline
              <select name="baseline">
                <option value="">No baseline comparison</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · {m.version.slice(0, 24)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label>
            Independent acceptance dataset
            <select name="dataset" required>
              <option value="">Choose independent dataset</option>
              {datasets.map((d) => (
                <option key={d.name}>{d.name}</option>
              ))}
            </select>
          </label>
          <details>
            <summary>Acceptance criteria</summary>
            <p>
              Bounds use nominal 95% Wilson intervals. Correlated detections can
              make these bounds overconfident; source diversity and domain
              review remain necessary.
            </p>
            <div className="learning-grid">
              {thresholds.map(([key, label, value, min, max, step]) => (
                <label key={key}>
                  {label}
                  <input
                    type="number"
                    name={key}
                    defaultValue={value}
                    min={min}
                    max={max}
                    step={step}
                    required
                  />
                </label>
              ))}
            </div>
          </details>
          <button
            className="primary"
            disabled={
              busy ||
              reports.some((r) =>
                ["queued", "running"].includes(r.payload.state),
              )
            }
          >
            Run acceptance evaluation
          </button>
        </fieldset>
      </form>
      <section className="panel padded">
        <h2>Acceptance reports</h2>
        {!reports.length && (
          <div className="learning-empty">
            No independent acceptance evaluations yet.
          </div>
        )}
        {reports.map((row) => {
          const p = row.payload,
            r = p.report;
          return (
            <article className="acceptance-report" key={row.id}>
              <div className="section-heading">
                <strong>
                  {models.find((m) => m.id === p.model_id)?.name || "Candidate"}{" "}
                  · {p.dataset}
                </strong>
                <span className="tag">
                  {r
                    ? r.passed
                      ? "Criteria passed"
                      : "Criteria failed"
                    : p.state}
                </span>
              </div>
              {p.error && <p className="learning-error">{p.error}</p>}
              {r && (
                <>
                  <div className="acceptance-metrics">
                    <span>
                      Samples <strong>{r.metrics.samples}</strong>
                    </span>
                    <span>
                      Source groups{" "}
                      <strong>{r.metrics.independent_groups}</strong>
                    </span>
                    <span>
                      {r.task === "detection" ? "AP50" : "Accuracy"}{" "}
                      <strong>
                        {(
                          (r.metrics.map50 ?? r.metrics.accuracy) * 100
                        ).toFixed(1)}
                        %
                      </strong>
                    </span>
                    <span>
                      Calibration error{" "}
                      <strong>
                        {r.metrics.ece?.toFixed(3) ?? "Unavailable"}
                      </strong>
                    </span>
                  </div>
                  {r.failures.length > 0 && (
                    <ul>
                      {r.failures.map((f: string) => (
                        <li key={f}>{f}</li>
                      ))}
                    </ul>
                  )}
                  <p>
                    {r.synthetic
                      ? "Synthetic validation only; production remains blocked."
                      : p.approved_by
                        ? "Acceptance approved. Activate a canary in the model registry before promotion."
                        : "Administrator approval is required before production eligibility."}
                  </p>
                  <details>
                    <summary>Metrics, policy and limitations</summary>
                    <pre>{JSON.stringify(r, null, 2)}</pre>
                  </details>
                  {r.deployment_eligible &&
                    !p.approved_by &&
                    user.role === "admin" && (
                      <button disabled={busy} onClick={() => approve(row.id)}>
                        Approve acceptance report
                      </button>
                    )}
                </>
              )}
            </article>
          );
        })}
      </section>
    </div>
  );
}
