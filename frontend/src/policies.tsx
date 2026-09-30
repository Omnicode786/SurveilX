import { useEffect, useState } from "react";

type Rule = {
  id: string;
  kind: string;
  labels: string[];
  zone: number[];
  direction: number[];
  duration: number;
  confidence: number;
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
        and does not measure speed. Video-event rules use calibrated clip scores
        and still require human review.
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
              <option value="scene_event">Video event class</option>
            </select>
          </label>
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
                  zone: [0, 0, 1, 1],
                  direction: [1, 0],
                  duration: 2,
                  confidence: 0.6,
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
