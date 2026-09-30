from dataclasses import dataclass
import math


@dataclass
class Candidate:
    camera_id: str
    priority: int
    age: float
    cost_ms: float
    available: bool = True


class Scheduler:
    """One global allocator. Deadline first, then operator priority, then age."""

    def __init__(self):
        self.credit_ms = 0.0

    def allocate(self, candidates, budget_ms, deadline_seconds, epoch_seconds=1.0):
        # Carry unused budget across a bounded coverage window. A slow but feasible
        # camera can reserve service instead of being deferred forever by fast ones.
        capacity = max(0, budget_ms) * max(1, math.ceil(deadline_seconds / max(0.05, epoch_seconds)))
        self.credit_ms = min(capacity, self.credit_ms + max(0, budget_ms))
        ordered = sorted(
            (c for c in candidates if c.available),
            key=lambda c: (
                c.age >= deadline_seconds,
                c.age if c.age >= deadline_seconds else c.priority,
                c.age,
            ),
            reverse=True,
        )
        selected, deferred = [], []
        available = remaining = self.credit_ms
        reservation = None
        for item in ordered:
            if reservation is None and item.cost_ms <= remaining:
                selected.append(item.camera_id)
                remaining -= item.cost_ms
            else:
                impossible = item.cost_ms > capacity
                if reservation is None and not impossible and item.age >= deadline_seconds:
                    reservation = item.camera_id
                deferred.append(
                    {
                        "camera_id": item.camera_id,
                        "coverage_violation": item.age >= deadline_seconds,
                        "infeasible": impossible,
                        "reason": (
                            "Measured cost exceeds the entire coverage-window budget; reduce workload"
                            if impossible
                            else "Accumulating service credit for an overdue camera"
                            if reservation
                            else "Measured cost exceeds available service credit"
                        ),
                    }
                )
        self.credit_ms = remaining
        return {
            "selected": selected,
            "deferred": deferred,
            "estimated_ms": available - remaining,
            "epoch_budget_ms": budget_ms,
            "available_credit_ms": available,
            "credit_remaining_ms": remaining,
            "reserved_for": reservation,
            "policy": "deadline-first-bounded-credit-v2",
        }
