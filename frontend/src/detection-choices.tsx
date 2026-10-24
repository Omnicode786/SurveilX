import { useEffect, useState } from "react";

type Option = {
  id: string;
  label: string;
  group: string;
  enabled: boolean;
  status: string;
};
type Choices = { disabled_labels: string[]; options: Option[] };
type Request = (path: string, method?: string, body?: unknown) => Promise<any>;
const order = [
  "People",
  "Fire and smoke",
  "Weapons",
  "Protective equipment and body parts",
  "Video events",
  "Alert rules",
  "Vehicles and traffic",
  "Animals",
  "Other objects",
];
export function DetectionChoices({
  cameraId,
  request,
  editable,
  onSaved,
}: {
  cameraId: string;
  request: Request;
  editable: boolean;
  onSaved: () => void;
}) {
  const [saved, setSaved] = useState<Choices | null>(null);
  const [disabled, setDisabled] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  useEffect(() => {
    let current = true;
    setSaved(null);
    setMessage("");
    request(`/cameras/${cameraId}/detection-options`)
      .then((value) => {
        if (current) {
          setSaved(value);
          setDisabled(value.disabled_labels);
        }
      })
      .catch((error) => current && setMessage(error.message));
    return () => {
      current = false;
    };
  }, [cameraId, request]);
  const toggle = (ids: string[], enabled: boolean) =>
    setDisabled((previous) =>
      enabled
        ? previous.filter((id) => !ids.includes(id))
        : [...new Set([...previous, ...ids])],
    );
  const dirty =
    saved &&
    JSON.stringify([...disabled].sort()) !==
      JSON.stringify([...saved.disabled_labels].sort());
  const save = async () => {
    setBusy(true);
    setMessage("");
    try {
      const value = await request(
        `/cameras/${cameraId}/detection-options`,
        "PUT",
        { disabled_labels: disabled },
      );
      setSaved(value);
      setDisabled(value.disabled_labels);
      setMessage("Detection choices saved.");
      onSaved();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="detection-choices" aria-label="Detection choices">
      <h3>What should this camera detect?</h3>
      <p>
        Turn off anything you do not want in new results or alerts. Existing
        recordings stay unchanged.
      </p>
      {!saved ? (
        <p role="status">{message || "Loading detection choices..."}</p>
      ) : (
        <>
          <div className="choices-toolbar">
            <label className="choice-search">
              Find an item
              <input
                type="search"
                placeholder="Search people, smoke, helmets..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>
            <button
              disabled={!editable || busy}
              onClick={() =>
                toggle(
                  saved.options.map((item) => item.id),
                  true,
                )
              }
            >
              Enable all
            </button>
            <button
              disabled={!editable || busy}
              onClick={() =>
                toggle(
                  saved.options.map((item) => item.id),
                  false,
                )
              }
            >
              Disable all
            </button>
          </div>
          <div className="choice-groups">
            {order.map((group) => {
              const all = saved.options.filter((item) => item.group === group);
              const items = all.filter((item) =>
                (item.label + " " + item.group.toLowerCase()).includes(
                  search.toLowerCase().trim(),
                ),
              );
              if (!items.length) return null;
              const count = all.filter(
                (item) => !disabled.includes(item.id),
              ).length;
              return (
                <div className="choice-group" key={group}>
                  <label className="choice-group-toggle">
                    <input
                      type="checkbox"
                      checked={count === all.length}
                      ref={(element) => {
                        if (element)
                          element.indeterminate =
                            count > 0 && count < all.length;
                      }}
                      disabled={!editable || busy}
                      onChange={(e) =>
                        toggle(
                          all.map((item) => item.id),
                          e.target.checked,
                        )
                      }
                    />
                    <span>
                      {group}
                      <small>
                        {count} of {all.length} on
                      </small>
                    </span>
                  </label>
                  <details open={search ? true : undefined}>
                    <summary>Choose individual items</summary>
                    <div className="choice-items">
                      {items.map((item) => (
                        <label key={item.id} className="choice-item">
                          <input
                            type="checkbox"
                            checked={!disabled.includes(item.id)}
                            disabled={!editable || busy}
                            onChange={(e) =>
                              toggle([item.id], e.target.checked)
                            }
                          />
                          <span>
                            {item.label}
                            <small>
                              {item.status === "rule"
                                ? "Configured alert rule"
                                : item.status === "no_rule"
                                  ? "Needs an alert rule"
                                  : item.status === "active"
                                    ? "Model active"
                                    : item.status === "saved_model"
                                      ? "Model saved; not active here"
                                      : "Needs a compatible model"}
                            </small>
                          </span>
                        </label>
                      ))}
                    </div>
                  </details>
                </div>
              );
            })}
            {!saved.options.some((item) =>
              (item.label + " " + item.group.toLowerCase()).includes(
                search.toLowerCase().trim(),
              ),
            ) && <p>No matching items.</p>}
          </div>
          <p className="choice-note">
            Unused models are skipped to save processing. Items sharing one
            model may have smaller speed gains. Turning an item on does not
            install or activate a model.
          </p>
          {editable ? (
            <div className="action-row choices-actions">
              <button
                className="primary"
                disabled={!dirty || busy}
                onClick={save}
              >
                {busy ? "Saving..." : "Save detection choices"}
              </button>
              <button
                disabled={!dirty || busy}
                onClick={() => {
                  setDisabled(saved.disabled_labels);
                  setMessage("");
                }}
              >
                Undo changes
              </button>
            </div>
          ) : (
            <p>Ask an administrator or operator to change these choices.</p>
          )}
          {message && <p role="status">{message}</p>}
        </>
      )}
    </section>
  );
}
