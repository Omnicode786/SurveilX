import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  Bell,
  Box,
  Camera,
  Check,
  ChevronRight,
  Cpu,
  Database,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Menu,
  Plus,
  Radio,
  Search,
  Settings,
  Shield,
  ShieldAlert,
  Terminal,
  Users,
  X,
} from "lucide-react";
import "./styles.css";
import { DetectionDetails, EventDetails, detectionSummary } from "./detections";
import { AcceptancePanel, AnnotationEditor, LearningPanel } from "./learning";
import { CapabilitiesPanel } from "./capabilities";
import { PolicyPanel } from "./policies";

type Row = Record<string, any>;
async function api(path: string, method = "GET", body?: unknown) {
  const response = await fetch("/api" + path, {
    method,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response
      .json()
      .catch(() => ({ detail: "Request failed" }));
    throw Object.assign(
      new Error(
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail),
      ),
      { status: response.status },
    );
  }
  return response.json();
}
const navigation = [
  ["overview", "Overview", LayoutDashboard],
  ["cameras", "Camera wall", Camera],
  ["incidents", "Incidents", ShieldAlert],
  ["alerts", "Alerts", Bell],
  ["analytics", "Analytics", Activity],
  ["controllers", "Controller", Radio],
  ["hardware", "Hardware", Cpu],
  ["models", "Model registry", Box],
  ["capabilities", "Task coverage", Shield],
  ["datasets", "Datasets", Database],
  ["experiments", "Experiments", FlaskConical],
  ["learning", "Reviewed learning", Activity],
  ["acceptance", "Model acceptance", Check],
  ["audit", "Audit trail", Terminal],
  ["users", "Team", Users],
  ["settings", "Settings", Settings],
] as const;
function Tag({
  children,
  tone = "",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={"tag " + tone}>{children}</span>;
}
function relative(time: number) {
  return time
    ? new Date(time * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—";
}
function App() {
  const [user, setUser] = useState<Row | null>(null),
    [ready, setReady] = useState(false),
    [page, setPage] = useState(location.pathname.split("/")[1] || "overview");
  const [status, setStatus] = useState<Row>({
      cameras: [],
      power: {},
      hardware: {},
    }),
    [rows, setRows] = useState<Row[]>([]),
    [incidents, setIncidents] = useState<Row[]>([]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [modal, setModal] = useState(""),
    [selected, setSelected] = useState<Row | null>(null),
    [menu, setMenu] = useState(false),
    [connected, setConnected] = useState(false),
    [filter, setFilter] = useState(""),
    [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  const [deploymentSlot, setDeploymentSlot] = useState("default");
  useEffect(() => {
    api("/auth/me")
      .then(setUser)
      .catch(() => {})
      .finally(() => setReady(true));
  }, []);
  const go = (next: string) => {
    setPage(next);
    history.pushState({}, "", "/" + next);
    setRows([]);
    setFilter("");
    setSelected(null);
    setMenu(false);
  };
  useEffect(() => {
    const pop = () => setPage(location.pathname.split("/")[1] || "overview");
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  const refresh = useCallback(async () => {
    if (!user) return;
    try {
      const [system, events] = await Promise.all([
        api("/controller/status"),
        api("/incidents"),
      ]);
      setStatus(system);
      setIncidents(events);
      const endpoint: Record<string, string> = {
        cameras: "/cameras",
        incidents: "/incidents",
        alerts: "/alerts",
        models: "/models",
        datasets: "/datasets",
        experiments: "/experiments",
        audit: "/audit",
        users: "/users",
        controllers: "/controller/decisions",
        analytics: "/hardware/metrics",
        settings: "/settings",
        hardware: "/hardware/status",
      };
      if (endpoint[page]) {
        const data = await api(endpoint[page]);
        setRows(Array.isArray(data) ? data : [data]);
      }
    } catch (e) {
      setError((e as Error).message);
      if ((e as Error & { status?: number }).status === 401) {
        setUser(null);
        setRows([]);
        setIncidents([]);
        setStatus({ cameras: [], power: {}, hardware: {} });
        setConnected(false);
      }
    }
  }, [user, page]);
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);
  useEffect(() => {
    if (!user) return;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    let closed = false;
    const connect = () => {
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/live/system`,
      );
      socket.onopen = () => setConnected(true);
      socket.onmessage = (e) => setStatus(JSON.parse(e.data).data);
      socket.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 3000);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [user]);
  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(timer);
  }, []);
  const perform = async (path: string, body?: unknown, method = "POST") => {
    setBusy(true);
    setError("");
    try {
      const result = await api(path, method, body);
      setNotice("Saved successfully");
      await refresh();
      return result;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 3500);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (!modal && !selected) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const controls = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          "button:not(:disabled),input,select,textarea,a[href]",
        ) || [],
      );
    controls()[0]?.focus();
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setModal("");
        setSelected(null);
      }
      if (event.key === "Tab") {
        const items = controls();
        if (!items.length) return;
        if (event.shiftKey && document.activeElement === items[0]) {
          event.preventDefault();
          items.at(-1)?.focus();
        } else if (!event.shiftKey && document.activeElement === items.at(-1)) {
          event.preventDefault();
          items[0].focus();
        }
      }
    };
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("keydown", keyboard);
      previous?.focus();
    };
  }, [modal, selected?.id]);
  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(event.currentTarget));
    if (modal === "import") {
      setBusy(true);
      setError("");
      try {
        const response = await fetch(
          `/api/datasets/${encodeURIComponent(String(values.name))}/import`,
          {
            method: "POST",
            body: values.bundle as File,
            headers: { "Content-Type": "application/zip" },
          },
        );
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Import failed");
        setModal("");
        setNotice("Dataset imported and validated");
        await refresh();
      } catch (error) {
        setError((error as Error).message);
      } finally {
        setBusy(false);
      }
      return;
    }
    let path = "",
      body: Row = {};
    if (modal === "camera") {
      path = "/cameras";
      body = {
        ...values,
        priority: Number(values.priority),
        zones: values.zone === "on" ? [[0.6, 0.2, 0.96, 0.9]] : [],
      };
      delete body.zone;
    }
    if (modal === "dataset") {
      path = "/datasets/generate";
      body = {
        name: values.name,
        count: Number(values.count),
        seed: Number(values.seed),
      };
    }
    if (modal === "training") {
      path = "/experiments";
      body = {
        dataset: values.dataset,
        epochs: Number(values.epochs),
        architecture: values.architecture,
      };
    }
    if (modal === "user") {
      path = "/users";
      body = values;
    }
    if (modal === "feedback") {
      path = "/feedback";
      body = { ...values, incident_id: selected?.id };
    }
    if (await perform(path, body)) setModal("");
  };
  if (!ready)
    return (
      <div className="loading">
        <Shield />
        Opening command center…
      </div>
    );
  if (!user)
    return (
      <div className="login">
        <div className="login-art">
          <span className="brand">
            <Shield /> SURVEIL<span>X</span>
          </span>
          <p className="eyebrow">ADAPTIVE EDGE INTELLIGENCE</p>
          <h1>
            See what matters.
            <br />
            <span>Act with context.</span>
          </h1>
          <p>
            A single workspace for your cameras, evidence, and the intelligence
            behind every decision.
          </p>
          <div className="login-grid">
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            try {
              setUser(
                await api(
                  "/auth/login",
                  "POST",
                  Object.fromEntries(new FormData(e.currentTarget)),
                ),
              );
              setError("");
            } catch (err) {
              setError((err as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <Tag tone="green">LOCAL COMMAND CENTER</Tag>
          <h2>Welcome back</h2>
          <p>Sign in to your SurveilX workspace.</p>
          <label>
            Username
            <input
              name="username"
              autoComplete="username"
              defaultValue="admin"
              required
            />
          </label>
          <label>
            Password
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              required
            />
          </label>
          {error && <div className="error">{error}</div>}
          <button className="primary" disabled={busy}>
            {busy ? "Signing in…" : "Enter command center"}
            <ArrowUpRight size={17} />
          </button>
          <small>
            First run? Your generated password is in
            <br />
            <code>data/initial-admin-password.txt</code>
          </small>
        </form>
      </div>
    );
  const cams: Row[] = status.cameras || [];
  const active = incidents.filter(
    (i) => !["RESOLVED", "FALSE_POSITIVE", "EXPIRED"].includes(i.state),
  );
  const title = navigation.find((n) => n[0] === page)?.[1] || "Overview";
  const visible = rows.filter((row) =>
    JSON.stringify(row).toLowerCase().includes(filter.toLowerCase()),
  );
  const cameraCard = (camera: Row) => (
    <button
      className="camera-card"
      key={camera.id}
      onClick={() => setSelected(camera)}
    >
      <div className="camera-image">
        {camera.status === "online" ? (
          <img
            src={`/api/cameras/${camera.id}/stream?t=${tick}`}
            alt={camera.name}
          />
        ) : (
          <div className="camera-off">
            <Camera />
            <span>{camera.status || "Waiting for source"}</span>
          </div>
        )}
        <div className="camera-overlay">
          <Tag tone={camera.status === "online" ? "green" : ""}>
            <i />
            {camera.status === "online" ? "LIVE" : camera.status}
          </Tag>
          {camera.synthetic && <Tag>GENERATED TEST</Tag>}
        </div>
        <div className="camera-bottom">
          <span>{camera.model || "Awaiting inference"}</span>
          <span>{Number(camera.latency_ms || 0).toFixed(1)} ms</span>
        </div>
      </div>
      <div className="camera-caption">
        <div>
          <strong>{camera.name}</strong>
          <small>
            {camera.environment} <span>·</span> Priority {camera.priority}
          </small>
        </div>
        <ArrowUpRight size={17} />
      </div>
      <div className="detected-caption">
        <span>DETECTED</span>
        <strong>
          {camera.status === "online"
            ? detectionSummary(camera.detections)
            : "Waiting for source"}
        </strong>
      </div>
    </button>
  );
  return (
    <div className="app">
      <aside className={menu ? "open" : ""}>
        <a
          className="brand"
          href="/overview"
          onClick={(e) => {
            e.preventDefault();
            go("overview");
          }}
        >
          <Shield /> SURVEIL<span>X</span>
          <small>EDGE</small>
        </a>
        <div className="workspace">
          <div className="workspace-mark">SX</div>
          <div>
            <strong>Local workspace</strong>
            <small>Adaptive intelligence</small>
          </div>
          <ChevronRight size={14} />
        </div>
        <p className="nav-label">COMMAND CENTER</p>
        <nav>
          {navigation.map(([id, label, Icon]) => (
            <React.Fragment key={id}>
              {id === "models" && <p className="nav-label">INTELLIGENCE LAB</p>}
              {id === "users" && <p className="nav-label">WORKSPACE</p>}
              <button
                className={page === id ? "active" : ""}
                onClick={() => go(id)}
              >
                <Icon size={17} />
                {label}
                {id === "incidents" && active.length > 0 && (
                  <b>{active.length}</b>
                )}
              </button>
            </React.Fragment>
          ))}
        </nav>
        <div className="side-bottom">
          <div className="engine-status">
            <i />
            <div>
              <strong>{connected ? "Live connection" : "Reconnecting"}</strong>
              <small>{status.power.level} compute profile</small>
            </div>
          </div>
          <button
            className="account"
            onClick={async () => {
              await api("/auth/logout", "POST");
              setUser(null);
            }}
          >
            <span className="avatar">
              {user.username.slice(0, 2).toUpperCase()}
            </span>
            <div>
              <strong>{user.username}</strong>
              <small>{user.role}</small>
            </div>
            <LogOut size={16} />
          </button>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <button
              className="mobile-menu icon-btn"
              aria-label="Toggle navigation"
              onClick={() => setMenu(!menu)}
            >
              <Menu />
            </button>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>{title}</strong>
          </div>
          <div>
            <Tag tone="green">
              <i />
              {connected ? "System connected" : "Polling active"}
            </Tag>
            <span className="date">
              {new Date().toLocaleDateString(undefined, {
                day: "2-digit",
                month: "short",
                year: "numeric",
              })}
            </span>
            <button
              className="icon-btn"
              aria-label="Open alerts"
              onClick={() => go("alerts")}
            >
              <Bell size={18} />
            </button>
          </div>
        </header>
        <section className="content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                SURVEILX /{" "}
                {page === "overview" ? "LIVE OPERATIONS" : "WORKSPACE"}
              </p>
              <h1>{page === "overview" ? "Operational overview" : title}</h1>
              <p>
                {(
                  {
                    overview:
                      "Your environment, in focus. Every observation. Every decision.",
                    cameras:
                      "Live sources, tracked entities, and the context behind them.",
                    incidents:
                      "Review the evidence. Keep every decision accountable.",
                    datasets:
                      "Versioned data for repeatable training and independent evaluation.",
                    hardware: "Compute that adapts to the machine you have.",
                    experiments:
                      "Train, evaluate, and calibrate. Keep every run reproducible.",
                  } as Row
                )[page] ||
                  "Measured evidence and explicit controls for your workspace."}
              </p>
            </div>
            {["overview", "cameras"].includes(page) &&
              user.role === "admin" && (
                <button className="primary" onClick={() => setModal("camera")}>
                  <Plus size={17} /> Add camera
                </button>
              )}
            {page === "datasets" && (
              <div className="action-row">
                <button onClick={() => setModal("import")}>
                  Import dataset
                </button>
                <button className="primary" onClick={() => setModal("dataset")}>
                  <Plus size={17} /> Generate dataset
                </button>
              </div>
            )}
            {page === "experiments" && (
              <button className="primary" onClick={() => setModal("training")}>
                <Plus size={17} /> Start training
              </button>
            )}
            {page === "users" && (
              <button className="primary" onClick={() => setModal("user")}>
                <Plus size={17} /> Add member
              </button>
            )}
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <Check size={16} />
              {notice}
            </div>
          )}
          {page === "overview" && (
            <>
              <div className="stats">
                <Stat
                  label="Connected cameras"
                  value={`${cams.filter((c) => c.status === "online").length} / ${cams.length}`}
                  icon={<Camera />}
                  detail="Sources reporting live frames"
                />
                <Stat
                  label="Awaiting review"
                  value={String(active.length).padStart(2, "0")}
                  icon={<ShieldAlert />}
                  detail="Observations requiring an operator"
                  warm
                />
                <Stat
                  label="Inference latency"
                  value={
                    status.mean_latency_ms == null
                      ? "—"
                      : `${status.mean_latency_ms.toFixed(1)}`
                  }
                  unit="ms"
                  icon={<Activity />}
                  detail="Measured mean · current session"
                />
                <Stat
                  label="Compute profile"
                  value={status.power.level || "economy"}
                  icon={<Cpu />}
                  detail="Automatically selected for this hardware"
                />
              </div>
              <div className="overview-layout">
                <div>
                  <div className="section-heading">
                    <h2>
                      Live camera wall <span>{cams.length} SOURCES</span>
                    </h2>
                    <button className="text-btn" onClick={() => go("cameras")}>
                      View all <ArrowUpRight size={15} />
                    </button>
                  </div>
                  <div className="camera-grid">
                    {cams.slice(0, 4).map(cameraCard)}
                  </div>
                  {!cams.length && (
                    <Empty
                      title="Bring your environment online"
                      text="Add a webcam, RTSP stream, local video, or generated test source."
                    />
                  )}
                  <div className="panel adaptation">
                    <div className="adaptation-icon">
                      <Cpu />
                    </div>
                    <div>
                      <h3>Intelligence, within your budget.</h3>
                      <p>
                        {status.power.reason ||
                          "Collecting hardware measurements."}
                      </p>
                      <small>
                        {status.power.resolution}px inference ·{" "}
                        {status.power.budget_ms} ms budget / epoch ·
                        CPU-compatible
                      </small>
                    </div>
                    <button
                      className="icon-btn"
                      aria-label="View hardware"
                      onClick={() => go("hardware")}
                    >
                      <ArrowUpRight />
                    </button>
                  </div>
                </div>
                <div>
                  <div className="section-heading">
                    <h2>Review queue</h2>
                    <Tag>{active.length} OPEN</Tag>
                  </div>
                  <div className="panel queue">
                    {active.slice(0, 5).map((event) => (
                      <button
                        className="queue-item"
                        key={event.id}
                        onClick={() => {
                          setSelected(event);
                          go("incidents");
                          setSelected(event);
                        }}
                      >
                        <div className="event-icon">
                          <ShieldAlert size={18} />
                        </div>
                        <div>
                          <strong>
                            {detectionSummary(event.details?.detections)}
                          </strong>
                          <small>Detected in monitored zone</small>
                          <small>
                            {relative(event.created)} ·{" "}
                            {event.state.toLowerCase().replaceAll("_", " ")}
                          </small>
                          <Tag tone="amber">Human review</Tag>
                        </div>
                        <ChevronRight size={14} />
                      </button>
                    ))}
                    {!active.length && (
                      <Empty
                        title="All clear in your queue"
                        text="New observations will appear here for review."
                      />
                    )}
                    <button
                      className="queue-footer"
                      onClick={() => go("incidents")}
                    >
                      Open incident center <ArrowUpRight size={15} />
                    </button>
                  </div>
                  <div className="panel resources">
                    <div className="section-heading">
                      <h2>Resource pulse</h2>
                      <i className="pulse" />
                    </div>
                    <Meter
                      label="CPU utilization"
                      value={status.hardware.cpu_percent}
                    />
                    <Meter
                      label="Memory utilization"
                      value={status.hardware.memory_percent}
                    />
                    <div className="resource-detail">
                      <span>Available memory</span>
                      <strong>
                        {status.hardware.available_gb?.toFixed(1) || "—"} GB
                      </strong>
                    </div>
                    <div className="resource-detail">
                      <span>GPU telemetry</span>
                      <strong>
                        {status.hardware.gpu?.name || "Unavailable"}
                      </strong>
                    </div>
                    <div className="resource-detail">
                      <span>Energy measurement</span>
                      <strong>
                        {status.hardware.gpu?.power_w != null
                          ? `${status.hardware.gpu.power_w} W`
                          : "Not measured"}
                      </strong>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
          {page === "cameras" && (
            <div className="camera-grid full">
              {visible.map(cameraCard)}
              {!visible.length && (
                <Empty
                  title="No camera sources"
                  text="Add a source to start monitoring."
                />
              )}
            </div>
          )}
          {page === "hardware" && (
            <>
              <div className="stats">
                <Stat
                  label="Processor cores"
                  value={String(status.hardware.cores || "—")}
                  detail={status.hardware.architecture || ""}
                  icon={<Cpu />}
                />
                <Stat
                  label="Installed memory"
                  value={status.hardware.ram_gb?.toFixed(1) || "—"}
                  unit="GB"
                  detail={status.hardware.os || ""}
                  icon={<Database />}
                />
                <Stat
                  label="Power level"
                  value={status.power.level}
                  detail="Application workload, not OS power plan"
                  icon={<Activity />}
                />
                <Stat
                  label="Temperature"
                  value={status.hardware.temperature_c ?? "—"}
                  unit="°C"
                  detail="Unavailable sensors remain unknown"
                  icon={<Radio />}
                />
              </div>
              <div className="panel padded">
                <h2>Acceleration capabilities</h2>
                <p>
                  Devices are eligible only when the installed runtime can
                  execute a compatible model. FPGA targets require a compiled
                  device-specific artifact.
                </p>
                <Json data={rows[0]?.accelerators || {}} />
                <h3>Adaptive policy</h3>
                <Json data={status.power} />
              </div>
            </>
          )}
          {page === "learning" && <LearningPanel request={api} user={user} />}
          {page === "capabilities" && <CapabilitiesPanel request={api} />}
          {page === "acceptance" && (
            <AcceptancePanel request={api} user={user} />
          )}
          {![
            "overview",
            "cameras",
            "hardware",
            "learning",
            "capabilities",
            "acceptance",
          ].includes(page) && (
            <>
              <div className="toolbar">
                <div className="search">
                  <Search size={16} />
                  <input
                    aria-label="Filter records"
                    placeholder={`Search ${title.toLowerCase()}…`}
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  />
                </div>
                <span>{visible.length} records</span>
              </div>
              {page === "settings" ? (
                <div className="panel padded">
                  <h2>Runtime policy</h2>
                  <p>
                    Configure these values in the local .env file, then restart
                    the service. All durations are seconds unless labeled
                    otherwise.
                  </p>
                  <Json data={rows[0] || {}} />
                </div>
              ) : (
                <div className="panel table-panel">
                  <table>
                    <thead>
                      <tr>
                        {(page === "incidents"
                          ? [
                              "Observation",
                              "Time",
                              "State",
                              "Evidence",
                              "Actions",
                            ]
                          : page === "datasets"
                            ? [
                                "Dataset",
                                "Domain",
                                "Samples",
                                "Source",
                                "Actions",
                              ]
                            : page === "users"
                              ? ["Username", "Role", "ID"]
                              : page === "alerts"
                                ? ["Incident", "Time", "Delivery", "Action"]
                                : ["Record", "Status / type", "Details"]
                        ).map((x) => (
                          <th key={x}>{x}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {visible.map((row, index) => (
                        <tr key={row.id || row.name || index}>
                          {page === "incidents" ? (
                            <>
                              <td>
                                <button
                                  className="row-link"
                                  onClick={() => setSelected(row)}
                                >
                                  {detectionSummary(row.details?.detections)}
                                  <ArrowUpRight size={14} />
                                </button>
                                <small>{row.id.slice(0, 8)}</small>
                              </td>
                              <td>{relative(row.created)}</td>
                              <td>
                                <Tag tone="amber">{row.state}</Tag>
                              </td>
                              <td>
                                {row.evidence_key ? (
                                  <a href={`/api/incidents/${row.id}/evidence`}>
                                    <ArrowDownToLine size={16} /> Bundle
                                  </a>
                                ) : (
                                  "Expired"
                                )}
                              </td>
                              <td>
                                <button onClick={() => setSelected(row)}>
                                  Review
                                </button>
                              </td>
                            </>
                          ) : page === "datasets" ? (
                            <>
                              <td>
                                <strong>{row.name}</strong>
                              </td>
                              <td>{row.domain}</td>
                              <td>{row.samples}</td>
                              <td>
                                <Tag tone={row.synthetic ? "amber" : "green"}>
                                  {row.synthetic ? "Synthetic" : "Imported"}
                                </Tag>
                              </td>
                              <td>
                                <button
                                  onClick={() =>
                                    perform(`/datasets/${row.name}/validate`)
                                  }
                                >
                                  Validate
                                </button>
                              </td>
                            </>
                          ) : page === "users" ? (
                            <>
                              <td>{row.username}</td>
                              <td>
                                <Tag>{row.role}</Tag>
                              </td>
                              <td>
                                <code>{row.id.slice(0, 8)}</code>
                              </td>
                            </>
                          ) : page === "alerts" ? (
                            <>
                              <td>
                                <code>{row.incident_id.slice(0, 8)}</code>
                              </td>
                              <td>{relative(row.created)}</td>
                              <td>
                                <Tag>{row.status}</Tag>
                              </td>
                              <td>
                                <button
                                  disabled={
                                    row.status === "acknowledged" || busy
                                  }
                                  onClick={() =>
                                    perform(`/alerts/${row.id}/acknowledge`)
                                  }
                                >
                                  Acknowledge
                                </button>
                              </td>
                            </>
                          ) : (
                            <>
                              <td>
                                <strong>
                                  {row.name ||
                                    row.payload?.result?.name ||
                                    row.payload?.dataset ||
                                    row.action ||
                                    row.payload?.action ||
                                    relative(row.timestamp) ||
                                    row.id}
                                </strong>
                                <small>{row.id?.slice(0, 8)}</small>
                              </td>
                              <td>
                                <Tag>
                                  {row.stage ||
                                    row.payload?.state ||
                                    row.kind ||
                                    row.actor ||
                                    "Recorded"}
                                </Tag>
                              </td>
                              <td>
                                <button onClick={() => setSelected(row)}>
                                  Inspect <ArrowUpRight size={13} />
                                </button>
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!visible.length && (
                    <Empty
                      title="No records yet"
                      text={
                        page === "models"
                          ? "Completed training runs register candidate models here."
                          : "Records will appear as your workspace is used."
                      }
                    />
                  )}
                </div>
              )}
              {page === "datasets" && (
                <div className="panel padded import-guide">
                  <h3>Bring your own training data</h3>
                  <p>
                    Place a versioned dataset in{" "}
                    <code>data/datasets/your-name/</code> with a{" "}
                    <code>manifest.json</code> and labeled images or normalized
                    NPZ clips. Define independent train, validation,
                    calibration, and test camera/session groups. The validator
                    checks shapes, checksums, taxonomy, and split leakage.
                  </p>
                  <p>
                    See <code>docs/data-and-training.md</code> for the import
                    contract. Generated test data verifies the workflow; it does
                    not establish real-world surveillance accuracy.
                  </p>
                </div>
              )}
            </>
          )}
          <footer>
            <span>
              <Shield size={13} /> SurveilX Edge · Research prototype
            </span>
            <span>Evidence over assumptions. Human review by design.</span>
          </footer>
        </section>
      </main>
      {selected && !modal && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <section
            className="dialog wide"
            role="dialog"
            aria-modal="true"
            aria-label="Record details"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="close icon-btn"
              aria-label="Close details"
              onClick={() => setSelected(null)}
            >
              <X />
            </button>
            <p className="eyebrow">EVIDENCE WORKSPACE</p>
            <h2>
              {selected.name ||
                selected.event_type?.replaceAll("_", " ") ||
                "Record details"}
            </h2>
            {selected.status && selected.name && (
              <img
                className="detail-image"
                src={`/api/cameras/${selected.id}/stream?t=${tick}`}
                alt={selected.name}
              />
            )}{" "}
            {selected.event_type && (
              <>
                <p>
                  This observation requires review before an incident decision.
                </p>
                <div className="action-row">
                  {["acknowledge", "resolve", "dismiss", "escalate"].map(
                    (action) => (
                      <button
                        key={action}
                        disabled={busy}
                        onClick={async () => {
                          const result = await perform(
                            `/incidents/${selected.id}/${action}`,
                          );
                          if (result) setSelected(result);
                        }}
                      >
                        {action}
                      </button>
                    ),
                  )}
                  <button onClick={() => setModal("feedback")}>
                    Add label
                  </button>
                  {user.role !== "viewer" && selected.evidence_key && (
                    <button onClick={() => setModal("annotation")}>
                      Annotate evidence
                    </button>
                  )}
                  {selected.evidence_key && (
                    <a
                      className="button"
                      href={`/api/incidents/${selected.id}/evidence`}
                    >
                      Download evidence
                    </a>
                  )}
                </div>
              </>
            )}
            {(selected.event_type || selected.detections) && (
              <DetectionDetails
                detections={selected.details?.detections || selected.detections}
                model={selected.details?.model || selected.model}
                calibration={
                  selected.details?.calibration || selected.calibration
                }
              />
            )}
            <EventDetails
              evidence={
                selected.details?.secondary_evidence ||
                selected.secondary_evidence
              }
            />
            {page === "cameras" && (
              <PolicyPanel
                cameraId={selected.id}
                request={api}
                editable={["admin", "operator"].includes(user.role)}
              />
            )}
            <details>
              <summary>Technical record</summary>
              <Json data={selected} />
            </details>
            {page === "models" && user.role === "admin" && (
              <div className="action-row">
                <label>
                  Specialist slot{" "}
                  <input
                    aria-label="Specialist slot"
                    value={deploymentSlot}
                    maxLength={48}
                    pattern="[a-zA-Z0-9_-]+"
                    onChange={(event) => setDeploymentSlot(event.target.value)}
                  />
                </label>
                <button
                  onClick={() => {
                    setSelected(null);
                    go("acceptance");
                  }}
                >
                  Evaluate acceptance
                </button>
                {selected.stage === "canary" &&
                  selected.manifest?.deployment_eligible && (
                    <button
                      disabled={busy}
                      onClick={async () => {
                        const result = await perform(
                          `/models/${selected.id}/deploy`,
                          {
                            stage: "production",
                            slot:
                              Object.entries(status.active_models || {})
                                .find(([, id]) => id === selected.id)?.[0]
                                .split(":")[1] || "default",
                          },
                        );
                        if (result) setSelected(result);
                      }}
                    >
                      Promote to production
                    </button>
                  )}
                <button
                  disabled={busy}
                  onClick={async () => {
                    const result = await perform(
                      `/models/${selected.id}/deploy`,
                      { stage: "canary", slot: deploymentSlot },
                    );
                    if (result) setSelected(result);
                  }}
                >
                  Activate scoped canary
                </button>
                <button
                  disabled={
                    busy || !["canary", "production"].includes(selected.stage)
                  }
                  onClick={async () => {
                    const result = await perform(
                      `/models/${selected.id}/rollback`,
                    );
                    if (result) setSelected({ ...selected, ...result });
                  }}
                >
                  Rollback
                </button>
              </div>
            )}
          </section>
        </div>
      )}
      {modal === "annotation" && selected && (
        <AnnotationEditor
          incident={selected}
          request={api}
          close={() => setModal("")}
          done={() => {
            setModal("");
            setSelected(null);
            setNotice("Annotation submitted for independent review.");
            go("learning");
          }}
        />
      )}
      {modal && modal !== "annotation" && (
        <div className="modal-backdrop">
          <form
            className="dialog"
            role="dialog"
            aria-modal="true"
            aria-label={modal}
            onSubmit={submit}
          >
            <button
              type="button"
              className="close icon-btn"
              aria-label="Close form"
              onClick={() => setModal("")}
            >
              <X />
            </button>
            <p className="eyebrow">WORKSPACE CONFIGURATION</p>
            <h2>
              {
                {
                  camera: "Connect a camera",
                  dataset: "Generate a test dataset",
                  import: "Import a dataset",
                  training: "Train a candidate",
                  user: "Add a team member",
                  feedback: "Label this observation",
                }[modal]
              }
            </h2>
            {modal === "import" && (
              <>
                <label>
                  Dataset version name
                  <input
                    name="name"
                    required
                    pattern="[a-zA-Z0-9_-]+"
                    placeholder="warehouse-v1"
                  />
                </label>
                <label>
                  Dataset bundle
                  <input name="bundle" type="file" accept=".zip" required />
                </label>
                <p>
                  ZIP containing manifest.json and NPZ clips. Maximum 100 MB
                  compressed. Validation checks grouped splits and labels before
                  import.
                </p>
              </>
            )}
            {modal === "camera" && (
              <>
                <label>
                  Camera name
                  <input name="name" required placeholder="North entrance" />
                </label>
                <label>
                  Source
                  <input
                    name="source"
                    required
                    placeholder="rtsp://… or demo://0"
                  />
                </label>
                <div className="form-row">
                  <label>
                    Environment
                    <input
                      name="environment"
                      list="environment-domains"
                      defaultValue="custom"
                      required
                      maxLength={100}
                    />
                    <datalist id="environment-domains">
                      {[
                        "custom",
                        "parking",
                        "office",
                        "warehouse",
                        "industrial",
                        "retail",
                        "traffic",
                        "building",
                      ].map((x) => (
                        <option key={x}>{x}</option>
                      ))}
                    </datalist>
                  </label>
                  <label>
                    Priority
                    <select name="priority">
                      {[1, 2, 3, 4, 5].map((x) => (
                        <option key={x}>{x}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <label className="check">
                  <input type="checkbox" name="zone" /> Monitor default
                  right-side test zone
                </label>
              </>
            )}
            {modal === "dataset" && (
              <>
                <label>
                  Dataset version name
                  <input
                    name="name"
                    required
                    pattern="[a-zA-Z0-9_-]+"
                    placeholder="generated-v1"
                  />
                </label>
                <div className="form-row">
                  <label>
                    Samples
                    <input
                      type="number"
                      name="count"
                      defaultValue="160"
                      min="80"
                      max="2000"
                      required
                    />
                  </label>
                  <label>
                    Seed
                    <input
                      type="number"
                      name="seed"
                      defaultValue="42"
                      min="0"
                      required
                    />
                  </label>
                </div>
                <p>
                  Creates labeled motion clips with four independent splits.
                  Intended for pipeline testing.
                </p>
              </>
            )}
            {modal === "training" && (
              <>
                <label>
                  Model architecture
                  <select name="architecture" defaultValue="auto">
                    <option value="auto">Automatic for dataset type</option>
                    <option value="scratch">
                      Custom SVA model from scratch
                    </option>
                    <option value="yolo_rai">
                      YOLO with scene/zone adapters
                    </option>
                  </select>
                </label>
                <label>
                  Dataset directory name
                  <input name="dataset" required placeholder="generated-v1" />
                </label>
                <label>
                  Epochs
                  <input
                    name="epochs"
                    type="number"
                    min="1"
                    max="500"
                    defaultValue="5"
                    required
                  />
                </label>
                <p>
                  Automatically validates data, trains, evaluates, calibrates,
                  and registers a candidate. Production deployment requires
                  separate acceptance.
                </p>
              </>
            )}
            {modal === "user" && (
              <>
                <label>
                  Username
                  <input name="username" required minLength={3} />
                </label>
                <label>
                  Password
                  <input
                    name="password"
                    type="password"
                    required
                    minLength={12}
                  />
                </label>
                <label>
                  Role
                  <select name="role">
                    {[
                      "viewer",
                      "operator",
                      "researcher",
                      "analyst",
                      "admin",
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                </label>
              </>
            )}
            {modal === "feedback" && (
              <>
                <label>
                  Label
                  <select name="label">
                    {[
                      "true_event",
                      "false_positive",
                      "ambiguous",
                      "missed_event",
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Review notes
                  <textarea name="notes" maxLength={4000} />
                </label>
                <p>
                  A second reviewer must approve labels before training use.
                </p>
              </>
            )}
            {error && <div className="error">{error}</div>}
            <button disabled={busy} className="primary" type="submit">
              {busy ? "Working…" : "Save and continue"}
              <ArrowUpRight size={17} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
function Stat({
  label,
  value,
  unit,
  detail,
  icon,
  warm = false,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  detail: string;
  icon: React.ReactNode;
  warm?: boolean;
}) {
  return (
    <div className={"stat " + (warm ? "warm" : "")}>
      <div>
        <span>{label}</span>
        {icon}
      </div>
      <strong>
        {value}
        <small>{unit}</small>
      </strong>
      <p>{detail}</p>
    </div>
  );
}
function Meter({ label, value = 0 }: { label: string; value?: number }) {
  return (
    <div className="meter">
      <div>
        <span>{label}</span>
        <strong>{value.toFixed(0)}%</strong>
      </div>
      <div className="meter-track">
        <i style={{ width: Math.min(100, value) + "%" }} />
      </div>
    </div>
  );
}
function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <Shield size={24} />
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}
function Json({ data }: { data: unknown }) {
  return <pre className="json">{JSON.stringify(data, null, 2)}</pre>;
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
