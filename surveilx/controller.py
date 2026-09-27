from dataclasses import dataclass


@dataclass
class Candidate:
    camera_id: str
    priority: int
    age: float
    cost_ms: float
    available: bool = True


class Scheduler:
    """One global allocator. Deadline first, then operator priority, then age."""

    def allocate(self, candidates, budget_ms, deadline_seconds):
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
        remaining = budget_ms
        for item in ordered:
            if item.cost_ms <= remaining:
                selected.append(item.camera_id)
                remaining -= item.cost_ms
            else:
                deferred.append(
                    {
                        "camera_id": item.camera_id,
                        "coverage_violation": item.age >= deadline_seconds,
                        "reason": "Measured cost exceeds remaining epoch budget",
                    }
                )
        return {
            "selected": selected,
            "deferred": deferred,
            "estimated_ms": budget_ms - remaining,
            "policy": "deadline-first-cold-start-v1",
        }
