# In-app notification policy

The Alerts page configures repeated-observation grouping, delivery cooldown, recipient roles, event priority overrides and acknowledgement deadlines. Administrators save the policy; other signed-in roles can inspect it. REST and the authenticated live feed apply the same recipient filter, and unrouted operators cannot acknowledge an alert.

Each new notification stores its policy snapshot and related incident IDs. Grouping preserves the incident records and their encrypted evidence. Incident grouping and notification cooldown are separate controls: one groups observations into an incident, the other groups notifications for the same camera and event. Different cameras and event types retain separate alerts. The existing incident grouping default remains 300 seconds; notification cooldown defaults to 60 seconds.

Timed escalation defaults to disabled (zero seconds). When enabled, the runtime checks persisted deadlines every ten scheduling cycles. A notification escalates once to the original escalation roles. It requests human attention and does not confirm the event or change an incident's state. Restarting the app preserves deadlines. Processing resumes when the local runtime worker runs; this is not a production delivery service.

A direct notification acknowledgement ends escalation. When all grouped incidents are acknowledged or in progress, incident reconciliation also acknowledges the notification; when all are resolved, false positives or expired, it closes it. Reviewing only one incident leaves the rest pending. Closed notifications reject acknowledgement. Policy edits affect future notifications; existing recipient snapshots and deadlines stay intact.

Delivery is persisted in-app and through the authenticated live feed. External email, SMS and webhooks are unconfigured, and no external message is sent. Disabling notifications preserves detection incidents and evidence and records the suppression in the audit log.

`tests/test_notifications.py` verifies grouped state, exact deadline handling, one-time escalation, acknowledgement, suppression, old-alert starvation protection, policy permissions and REST/live routing. The full local suite passed 174 tests on 2026-10-04.
