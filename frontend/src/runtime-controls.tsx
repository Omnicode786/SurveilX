import { useEffect, useState } from "react";

type Request = (path: string, method?: string, body?: unknown) => Promise<any>;
export function RuntimeControls({
  request,
  admin,
}: {
  request: Request;
  admin: boolean;
}) {
  const [profile, setProfile] = useState("auto"),
    [scale, setScale] = useState(1);
  const [models, setModels] = useState<any[]>([]),
    [cameras, setCameras] = useState<any[]>([]);
  const [model, setModel] = useState(""),
    [camera, setCamera] = useState("");
  const [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    Promise.all([request("/settings"), request("/models"), request("/cameras")])
      .then(([policy, available, sources]) => {
        setProfile(policy.power_profile);
        setScale(policy.runtime_budget_scale);
        setModels(available);
        setCameras(sources);
      })
      .catch((error) => setMessage(String(error.message)));
  }, [request]);
  const selected = models.find((item) => item.id === model);
  const run = async (action: () => Promise<unknown>, success: string) => {
    setBusy(true);
    setMessage("");
    try {
      await action();
      setMessage(success);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="panel padded">
      <h2>Performance</h2>
      <p>
        Choose how much processing power SurveilX can use. Automatic is a good
        starting point.
      </p>
      <div className="action-row">
        <label>
          Performance mode{" "}
          <select
            aria-label="Performance mode"
            value={profile}
            disabled={!admin || busy}
            onChange={(e) => setProfile(e.target.value)}
          >
            <option value="auto">Automatic</option>
            <option value="economy">Economy</option>
            <option value="balanced">Balanced</option>
            <option value="performance">Performance</option>
          </select>
        </label>
        <details className="advanced-options">
          <summary>Advanced performance</summary>{" "}
          <label>
            Inference budget multiplier{" "}
            <input
              aria-label="Inference budget multiplier"
              type="number"
              min="0.25"
              max="2"
              step="0.25"
              value={scale}
              disabled={!admin || busy}
              onChange={(e) => setScale(Number(e.target.value))}
            />
          </label>
        </details>
        <button
          disabled={!admin || busy}
          onClick={() =>
            run(
              () =>
                request("/settings/runtime", "PATCH", {
                  power_profile: profile,
                  runtime_budget_scale: scale,
                }),
              "Performance preference saved; active workers adapt automatically.",
            )
          }
        >
          Save performance mode
        </button>
      </div>
      <details className="advanced-options">
        <summary>Advanced: choose a model</summary>
        <h3>Test a saved model</h3>
        <p>
          Select a model explicitly for local review. This does not grant
          production approval. Cameras must use the model's domain; existing
          production gates and rollback remain in Model registry.
        </p>
        <label>
          Model{" "}
          <select
            aria-label="Runtime model"
            value={model}
            disabled={!admin || busy}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="">Select a registered model</option>
            {models.map((item) => (
              <option key={item.id} value={item.id}>
                {item.version} - {item.manifest?.domain} - {item.stage}
              </option>
            ))}
          </select>
        </label>
        {selected && (
          <p>
            Classes: {(selected.manifest?.classes || []).join(", ")}. Task:{" "}
            {selected.manifest?.task}. Domain: {selected.manifest?.domain}.
          </p>
        )}
        <label>
          Assign camera to selected model domain{" "}
          <select
            aria-label="Model camera"
            value={camera}
            disabled={!admin || busy}
            onChange={(e) => setCamera(e.target.value)}
          >
            <option value="">Keep current camera domains</option>
            {cameras.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} - {item.environment}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={!admin || busy || !model}
          onClick={() =>
            run(async () => {
              await request(`/models/${model}/deploy`, "POST", {
                stage: "canary",
                slot: "admin-selection",
              });
              if (camera)
                await request(`/cameras/${camera}`, "PATCH", {
                  environment: selected.manifest.domain,
                });
            }, "Scoped canary selected. Live workers must be running; production approval is unchanged.")
          }
        >
          Activate development canary
        </button>
      </details>
      {!admin && <p>Only an administrator can change these controls.</p>}
      {message && <p role="status">{message}</p>}
    </div>
  );
}
