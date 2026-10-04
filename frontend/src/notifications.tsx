import { useEffect, useState } from "react";

type Policy = {
  enabled: boolean;
  incident_group_seconds: number;
  cooldown_seconds: number;
  acknowledgement_seconds: number;
  recipient_roles: string[];
  escalation_roles: string[];
  event_priorities: Record<string, string>;
};

export function NotificationSettings({
  request,
  user,
}: {
  request: (path: string, method?: string, body?: unknown) => Promise<any>;
  user: { role: string };
}) {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [eventName, setEventName] = useState("");
  const editable = user.role === "admin";
  useEffect(() => {
    let cancelled = false;
    request("/notifications/policy")
      .then((value) => {
        if (!cancelled) setPolicy(value);
      })
      .catch((error) => {
        if (!cancelled) setMessage(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [request]);
  if (!policy)
    return <p role="status">{message || "Loading notification policy…"}</p>;
  async function save() {
    setBusy(true);
    try {
      const value = await request("/notifications/policy", "PUT", policy);
      setPolicy(value);
      setMessage(
        "Saved. Applies to new notifications; existing deadlines keep their original policy.",
      );
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function roles(
    field: "recipient_roles" | "escalation_roles",
    role: string,
    enabled: boolean,
  ) {
    if (!policy) return;
    const values = enabled
      ? [...policy[field], role]
      : policy[field].filter((value) => value !== role);
    setPolicy({ ...policy, [field]: [...new Set(values)] });
  }
  return (
    <details className="panel padded learning-form">
      <summary>Notification policy</summary>
      <p>
        Group repeated observations, route alerts by role, and escalate
        notifications that have not been acknowledged. Escalation requests
        review of the detected event.
      </p>
      <label>
        <input
          type="checkbox"
          checked={policy.enabled}
          disabled={!editable || busy}
          onChange={(event) =>
            setPolicy({ ...policy, enabled: event.target.checked })
          }
        />{" "}
        Enable in-app notifications
      </label>
      <div className="form-grid">
        {(
          [
            ["incident_group_seconds", "Incident grouping window", 3600],
            ["cooldown_seconds", "Notification cooldown", 3600],
            ["acknowledgement_seconds", "Escalate after; zero disables", 86400],
          ] as const
        ).map(([field, title, maximum]) => (
          <label key={field}>
            {title} (seconds)
            <input
              type="number"
              min={0}
              max={maximum}
              value={policy[field]}
              disabled={!editable || busy}
              onChange={(event) =>
                setPolicy({ ...policy, [field]: Number(event.target.value) })
              }
            />
          </label>
        ))}
      </div>
      {(
        [
          ["recipient_roles", "Notify roles"],
          ["escalation_roles", "Escalate to roles"],
        ] as const
      ).map(([field, title]) => (
        <fieldset key={field} disabled={!editable || busy}>
          <legend>{title}</legend>
          {["admin", "operator", "researcher", "analyst", "viewer"].map(
            (role) => (
              <label key={role}>
                <input
                  type="checkbox"
                  checked={policy[field].includes(role)}
                  onChange={(event) => roles(field, role, event.target.checked)}
                />{" "}
                {role}
              </label>
            ),
          )}
        </fieldset>
      ))}
      <fieldset disabled={!editable || busy}>
        <legend>Event priority overrides</legend>
        <p>
          Use the event identifier shown on an alert. Other events retain their
          incident severity.
        </p>
        {Object.entries(policy.event_priorities).map(([event, priority]) => (
          <div className="action-row" key={event}>
            <code>{event}</code>
            <select
              aria-label={`Priority for ${event}`}
              value={priority}
              onChange={(change) =>
                setPolicy({
                  ...policy,
                  event_priorities: {
                    ...policy.event_priorities,
                    [event]: change.target.value,
                  },
                })
              }
            >
              {["review", "low", "medium", "high", "critical"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
            <button
              onClick={() => {
                const priorities = { ...policy.event_priorities };
                delete priorities[event];
                setPolicy({ ...policy, event_priorities: priorities });
              }}
            >
              Remove
            </button>
          </div>
        ))}
        {editable && (
          <div className="action-row">
            <input
              aria-label="Event identifier"
              placeholder="Event identifier"
              maxLength={80}
              value={eventName}
              onChange={(event) => setEventName(event.target.value)}
            />
            <button
              disabled={
                !eventName.trim() ||
                Object.keys(policy.event_priorities).length >= 64
              }
              onClick={() => {
                setPolicy({
                  ...policy,
                  event_priorities: {
                    ...policy.event_priorities,
                    [eventName.trim()]: "review",
                  },
                });
                setEventName("");
              }}
            >
              Add event
            </button>
          </div>
        )}
      </fieldset>
      <p>Delivery is in-app and through the authenticated live alert feed.</p>
      {editable && (
        <button
          disabled={
            busy ||
            !policy.recipient_roles.length ||
            !policy.escalation_roles.length
          }
          onClick={save}
        >
          {busy ? "Saving…" : "Save notification policy"}
        </button>
      )}
      {message && <p role="status">{message}</p>}
    </details>
  );
}
