"""Group class routes by approved loaded expert, with bounded retries and recovery."""

import time
from surveilx.model_selection import rank_categories


class CategoryRouter:
    name = "approved-category-router"

    def __init__(self, models, maximum_models=3):
        self.models = models
        self.maximum_models = maximum_models
        self.failed_until = {}
        self.failures = {}
        self.cursor = 0
        self.routes = []
        self.calibrated = True
        self.version = "|".join(sorted(m["version"] for m in models))
        self.by_version = {m["version"]: m["expert"] for m in models}
        recommendations = rank_categories(models)
        # Conflicting evaluation corpora cannot be fairly ranked against one another.
        classes = {}
        for recommendation in recommendations:
            classes.setdefault(recommendation["category"], []).append(recommendation)
        if any(len(choices) > 1 for choices in classes.values()):
            self.last = {}
            return  # preserve explicit slot scheduling rather than compare incompatible corpora
        for category, choices in classes.items():
            if len(choices) != 1:
                continue
            choice = choices[0]
            self.routes.append(
                {
                    "category": category,
                    "candidates": [
                        choice["selected"]["version"],
                        *[m["version"] for m in choice["fallbacks"]],
                    ],
                }
            )
        self.last = {}

    def infer(self, frame, resolution=320):
        now = time.monotonic()
        outputs = []
        cache = {}
        used = []
        # Rotate category start only when constrained; shared experts still execute once.
        ordered = self.routes[self.cursor :] + self.routes[: self.cursor]
        self.cursor = (self.cursor + 1) % max(1, len(self.routes))
        route_status = []
        for route in ordered:
            selected = None
            for version in route["candidates"]:
                if self.failed_until.get(version, 0) > now:
                    continue
                if version not in cache:
                    if len(used) >= self.maximum_models:
                        continue
                    used.append(version)
                    start = time.perf_counter()
                    try:
                        # Ranking evidence is for the artifact's native calibrated geometry.
                        cache[version] = self.by_version[version].infer(frame, None)
                        self.failures.pop(version, None)
                        self.last[version] = {
                            "latency_ms": (time.perf_counter() - start) * 1000,
                            "state": "healthy",
                        }
                    except Exception as exc:
                        self.failures[version] = self.failures.get(version, 0) + 1
                        self.failed_until[version] = now + min(60, 5 * 2 ** min(self.failures[version], 4))
                        self.last[version] = {
                            "state": "failed",
                            "error": type(exc).__name__,
                            "retry_in_seconds": self.failed_until[version] - now,
                        }
                        cache[version] = None
                if cache[version] is not None:
                    outputs.extend(d for d in cache[version] if d.label == route["category"])
                    selected = version
                    break
            route_status.append(
                {
                    "category": route["category"],
                    "version": selected,
                    "state": "served" if selected else "deferred_or_failed",
                }
            )
        self.last_routes = route_status
        if self.routes and not any(row["state"] == "served" for row in route_status):
            raise RuntimeError("All eligible routed experts failed; use explicit baseline fallback")
        return outputs

    def status(self):
        return {
            "routes": getattr(self, "last_routes", self.routes),
            "models": dict(self.last),
            "maximum_models_per_frame": self.maximum_models,
            "recovery": "exponential retry cooldown, capped60s",
        }
