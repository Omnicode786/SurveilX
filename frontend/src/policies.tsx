import { useEffect, useState } from "react";

type Rule = {
  id: string;
  kind: string;
  labels: string[];
  secondary_labels: string[];
  zone: number[];
  signal_zone: number[];
  stop_line: number[];
  direction: number[];
  ground_plane_transform: number[];
  calibration_id: string | null;
  duration: number;
  confidence: number;
  max_distance: number;
  speed_mps: number;
};
export function PolicyPanel({
  cameraId,
  request,
  editable,
}: {
  cameraId: string;
  request: (path: string, method?: string, body?: unknown) => Promise<any>;
  editable: boolean;
}) {
  const [rules, setRules] = useState<Rule[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let mounted = true;
    request(`/cameras/${cameraId}/policies`)
      .then((value) => {
        if (mounted) setRules(value.rules);
      })
      .catch((error) => {
        if (mounted) setMessage(error.message);
      });
    return () => {
      mounted = false;
    };
  }, [cameraId, request]);
  function change(index: number, update: Partial<Rule>) {
    setRules((previous) =>
      previous.map((rule, i) => (i === index ? { ...rule, ...update } : rule)),
    );
  }
  return (
    <section className="panel padded">
      <h2>Incident rules</h2>
      <p>
        Rules create review candidates after repeated observations. They require
        an active model with the relevant labels. Missing-helmet review requires
        a visible, unambiguous head; traffic direction uses image coordinates
        and calibrated speed uses site geometry. Missing-vest review requires a
        separate, unambiguous torso detection. Proximity is an image-coordinate
        threshold, not physical distance. Signal, machine-state and blocked-exit
        rules use configured labels and geometry. Video-event rules use
        calibrated clip scores and still require human review.
      </p>
      {rules.map((rule, index) => (
        <fieldset key={index} disabled={!editable || busy}>
          <label>
            Rule name
            <input
              value={rule.id}
              onChange={(e) => change(index, { id: e.target.value })}
            />
          </label>
          <label>
            Observation
            <select
              value={rule.kind}
              onChange={(e) => change(index, { kind: e.target.value })}
            >
              <option value="object_presence">
                Object presence (such as fire or smoke)
              </option>
              <option value="restricted_zone">Object in restricted area</option>
              <option value="wrong_way">
                Travel against configured direction
              </option>
              <option value="possible_missing_helmet">
                Possible missing helmet
              </option>
              <option value="possible_missing_vest">
                Possible missing safety vest
              </option>
              <option value="configured_proximity">
                Tracked objects within image distance
              </option>
              <option value="calibrated_speed">
                Tracked speed from site calibration
              </option>
              <option value="signal_stop_line">
                Stop-line crossing with signal state
              </option>
              <option value="machine_state">
                Machine with configured state label
              </option>
              <option value="blocked_exit">Blocked exit or access zone</option>
              <option value="scene_event">Video event class</option>
            </select>
          </label>
          {["configured_proximity", "signal_stop_line", "machine_state"].includes(
            rule.kind,
          ) && (
            <>
              <label>
                Secondary labels, comma separated
                <input
                  value={(rule.secondary_labels || []).join(",")}
                  onChange={(e) =>
                    change(index, {
                      secondary_labels: e.target.value
                        .split(",")
                        .map((v) => v.trim()),
                    })
                  }
                />
              </label>
            </>
          )}
          {["configured_proximity", "machine_state"].includes(rule.kind) && (
            <>
              <label>
                Maximum normalized center distance
                <input
                  type="number"
                  min="0.01"
                  max="2"
                  step="0.01"
                  value={rule.max_distance ?? 0.15}
                  onChange={(e) =>
                    change(index, { max_distance: Number(e.target.value) })
                  }
                />
              </label>
            </>
          )}
          <label>
            Object or event labels, comma separated
            <input
              value={rule.labels.join(",")}
              onChange={(e) =>
                change(index, {
                  labels: e.target.value.split(",").map((v) => v.trim()),
                })
              }
            />
          </label>
          <div className="action-row">
            {["Left", "Top", "Right", "Bottom"].map((label, i) => (
              <label key={label}>
                {label}
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={rule.zone[i]}
                  onChange={(e) =>
                    change(index, {
                      zone: rule.zone.map((v, j) =>
                        i === j ? Number(e.target.value) : v,
                      ),
                    })
                  }
                />
              </label>
            ))}
          </div>
          {rule.kind === "wrong_way" && (
            <div className="action-row">
              {[
                "Allowed horizontal direction",
                "Allowed vertical direction",
              ].map((label, i) => (
                <label key={label}>
                  {label}
                  <input
                    type="number"
                    step="0.1"
                    value={rule.direction[i]}
                    onChange={(e) =>
                      change(index, {
                        direction: rule.direction.map((v, j) =>
                          i === j ? Number(e.target.value) : v,
                        ),
                      })
                    }
                  />
                </label>
              ))}
            </div>
          )}
          {rule.kind === "signal_stop_line" && (
            <>
              <div className="action-row">
                {["Signal left", "Signal top", "Signal right", "Signal bottom"].map(
                  (label, i) => (
                    <label key={label}>
                      {label}
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={(rule.signal_zone || [0, 0, 1, 1])[i]}
                        onChange={(e) =>
                          change(index, {
                            signal_zone: (rule.signal_zone || [0, 0, 1, 1]).map(
                              (v, j) => (i === j ? Number(e.target.value) : v),
                            ),
                          })
                        }
                      />
                    </label>
                  ),
                )}
              </div>
              <div className="action-row">
                {["Line x1", "Line y1", "Line x2", "Line y2"].map(
                  (label, i) => (
                    <label key={label}>
                      {label}
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={(rule.stop_line || [0.5, 0, 0.5, 1])[i]}
                        onChange={(e) =>
                          change(index, {
                            stop_line: (rule.stop_line || [0.5, 0, 0.5, 1]).map(
                              (v, j) => (i === j ? Number(e.target.value) : v),
                            ),
                          })
                        }
                      />
                    </label>
                  ),
                )}
              </div>
            </>
          )}
          {rule.kind === "calibrated_speed" && (
            <>
              <label>
                Site calibration ID
                <input
                  value={rule.calibration_id || ""}
                  onChange={(e) =>
                    change(index, { calibration_id: e.target.value || null })
                  }
                />
              </label>
              <label>
                Speed threshold, m/s
                <input
                  type="number"
                  min="0.1"
                  max="120"
                  step="0.1"
                  value={rule.speed_mps ?? 5}
                  onChange={(e) =>
                    change(index, { speed_mps: Number(e.target.value) })
                  }
                />
              </label>
              <div className="action-row">
                {(rule.ground_plane_transform || [1, 0, 0, 0, 1, 0, 0, 0, 1]).map(
                  (value, transformIndex) => (
                    <label key={transformIndex}>
                      H{transformIndex + 1}
                      <input
                        type="number"
                        step="0.0001"
                        value={value}
                        onChange={(e) =>
                          change(index, {
                            ground_plane_transform: (
                              rule.ground_plane_transform || [
                                1, 0, 0, 0, 1, 0, 0, 0, 1,
                              ]
                            ).map((v, j) =>
                              transformIndex === j ? Number(e.target.value) : v,
                            ),
                          })
                        }
                      />
                    </label>
                  ),
                )}
              </div>
            </>
          )}
          <label>
            Persistence in seconds
            <input
              type="number"
              min="0.5"
              max="60"
              step="0.5"
              value={rule.duration}
              onChange={(e) =>
                change(index, { duration: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Minimum detection score
            <input
              type="number"
              min="0.1"
              max="1"
              step="0.05"
              value={rule.confidence}
              onChange={(e) =>
                change(index, { confidence: Number(e.target.value) })
              }
            />
          </label>
          <button
            type="button"
            onClick={() => setRules(rules.filter((_, i) => i !== index))}
          >
            Remove rule
          </button>
        </fieldset>
      ))}
      {editable && (
        <div className="action-row">
          <button
            disabled={busy || rules.length >= 16}
            onClick={() =>
              setRules([
                ...rules,
                {
                  id: `rule_${Date.now()}`,
                  kind: "object_presence",
                  labels: ["person"],
                  secondary_labels: [],
                  zone: [0, 0, 1, 1],
                  signal_zone: [0, 0, 1, 1],
                  stop_line: [0.5, 0, 0.5, 1],
                  direction: [1, 0],
                  ground_plane_transform: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                  calibration_id: null,
                  duration: 2,
                  confidence: 0.6,
                  max_distance: 0.15,
                  speed_mps: 5,
                },
              ])
            }
          >
            Add rule
          </button>
          <button
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                const result = await request(
                  `/cameras/${cameraId}/policies`,
                  "PUT",
                  { rules },
                );
                setRules(result.rules);
                setMessage(
                  "Rules saved. Camera observation history restarted.",
                );
              } catch (error) {
                setMessage((error as Error).message);
              } finally {
                setBusy(false);
              }
            }}
          >
            Save rules
          </button>
        </div>
      )}
      {message && <p role="status">{message}</p>}
    </section>
  );
}
