# System diagrams

These diagrams distinguish executable paths from research/deployment targets. The complete runtime overview is in [architecture.md](architecture.md).

## 1. Overall system

```mermaid
flowchart LR
  Cameras --> Capture --> State --> Control --> Experts --> Evidence
  Evidence --> Incidents --> Dashboard --> Feedback --> OfflineTraining
  OfflineTraining --> Registry --> Experts
  Hardware --> Control
```

## 2. RAIC

```mermaid
flowchart TD
  Observation[Frame age, context and measured costs] --> Eligible{Expert available?}
  Eligible -->|No| Defer[Record unavailable action]
  Eligible -->|Yes| Budget[Apply hardware budget]
  Budget --> Candidates[Produce feasible action candidates]
  Candidates --> Scheduler[MCRS]
```

## 3. ASIE

```mermaid
flowchart LR
  Operator[Configured environment and zones] --> Profile[Versioned scene profile]
  Frame --> Quality[Brightness and motion]
  Quality --> State[Shared camera state]
  Profile --> State
  State --> Controller
```

Automatic semantic scene classification is not implemented; the diagram accurately shows the manual-profile cold start.

## 4. YOLO-RAI research target

```mermaid
flowchart TD
  Frame --> Base[Supplied YOLO detector]
  Context -.-> Adapter[Experimental risk conditioner]
  Zones -.-> Adapter
  Adapter -.-> Base
  Base --> Boxes[Boxes and raw scores]
  Boxes --> Heldout[Held-out calibration required]
```

Dashed conditioning edges are not yet integrated/trained in a YOLO backbone.

## 5. SVA-Net

```mermaid
flowchart TD
  Clip --> Encoder[Shared spatial CNN]
  Encoder --> Entities[Box-center entity sampling]
  Boxes --> Entities
  Boxes --> Trajectory[Explicit trajectory differences]
  Trajectory --> Motion[Motion projection]
  Entities --> Motion
  Motion --> Temporal[Depthwise temporal convolution]
  Temporal --> Context[Context affine adapter]
  Context --> Relations[Pairwise relation messages]
  Relations --> Event[Event head]
  Relations --> Other[Entity and relation heads: untrained]
```

## 6. Arbitration

```mermaid
flowchart TD
  Evidence --> Calibration{Calibrated?}
  Calibration -->|No| Abstain
  Calibration -->|Yes| Scope{Same task, domain and taxonomy?}
  Scope -->|No| Abstain
  Scope -->|Yes| Conflict{Material disagreement?}
  Conflict -->|Yes| Review[Human review]
  Conflict -->|No| Pool[Compatible evidence pooling]
```

## 7. MCRS

```mermaid
flowchart LR
  Ready[Ready cameras] --> Deadlines[Coverage age ordering]
  Deadlines --> Priorities[Operator priority for non-overdue cameras]
  Priorities --> Allocate[Fit measured cost into global budget]
  Allocate --> Execute
  Allocate --> Infeasible[Log deferred work and missed coverage]
```

## 8. Incident lifecycle

```mermaid
stateDiagram-v2
  [*] --> VERIFYING: Persistent zone observation
  VERIFYING --> CONFIRMED: Operator verification
  CONFIRMED --> ALERTED
  VERIFYING --> ACKNOWLEDGED
  ALERTED --> ACKNOWLEDGED
  ACKNOWLEDGED --> IN_PROGRESS
  ACKNOWLEDGED --> RESOLVED
  IN_PROGRESS --> RESOLVED
  VERIFYING --> FALSE_POSITIVE
  VERIFYING --> ESCALATED
  ESCALATED --> ACKNOWLEDGED
  VERIFYING --> EXPIRED
```

The full transition table in `surveilx/incidents.py` is authoritative. A delivered review notification does not automatically mean the event is confirmed.

## 9. Admin architecture

```mermaid
flowchart TD
  Session[Authenticated session] --> Shell[Responsive React command center]
  Shell --> Operations[Overview, cameras, incidents, alerts]
  Shell --> Intelligence[Controller, hardware, models]
  Shell --> Research[Datasets and experiments]
  Shell --> Governance[Users, settings and audit]
  API --> Shell
  WebSocket --> Shell
```

## 10. Notification architecture

```mermaid
flowchart LR
  Incident --> Group[Group existing open incident]
  Group --> New{New incident?}
  New -->|Yes| Alert[Persist in-app alert]
  Alert --> Dashboard
  Alert --> WebSocket
  Dashboard --> Acknowledge[Operator acknowledgment]
  Acknowledge --> Audit
  Alert -.-> External[Future configured email/SMS/webhook delivery workers]
```

External delivery retries, recipient routing and escalation timers remain implementation work; no external messages have been sent.

## 11. Continuous learning

```mermaid
flowchart LR
  Feedback --> SecondReviewer --> Annotation[Clip annotation and dataset version]
  Annotation --> Split[Independent grouped splits]
  Split --> Train --> Validate --> Calibrate --> Test
  Test --> Candidate[Candidate registry]
  Candidate --> Canary[Explicit local activation]
  Canary -.-> Production[Real-domain acceptance required]
  Production --> Rollback
```

## 12. Deployment

```mermaid
flowchart TD
  Browser --> API[One FastAPI worker]
  API --> DB[(PostgreSQL deployment / SQLite local)]
  API --> Objects[Encrypted filesystem or S3 objects]
  API --> Capture[Concurrent source capture]
  API --> Compute[Global inference runtime]
  Compute --> CPU[Normal PC / native CPU kernels]
  Compute -.-> GPU[CUDA / compatible runtime]
  Compute -.-> FPGA[Board-specific VART or ONNX provider]
  Compute -.-> Edge[Jetson / Pi target validation]
```

## 13. Database

```mermaid
erDiagram
  users ||--o{ sessions : authenticates
  cameras ||--o{ incidents : observes
  incidents ||--o| alerts : notifies
  incidents ||--o{ feedback_labels : reviewed
  users ||--o{ audit_logs : acts
  model_versions ||--o{ runtime_records : referenced_by_payload
```

The final relationship is a logical reference in telemetry JSON, not an enforced foreign key. Multi-organization/site normalization and high-volume time-series partitioning are not implemented.
