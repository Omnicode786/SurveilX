"""Explicit task contracts; a declared profile is not evidence of a trained capability."""

DOMAINS = ["parking", "office", "warehouse", "industrial", "retail", "traffic", "building"]


def profile(key, title, task, classes, domains, interpretation, requirements):
    return {
        "id": key,
        "title": title,
        "task": task,
        "required_classes": classes,
        "domains": domains,
        "interpretation": interpretation,
        "requirements": requirements,
    }


PROFILES = [
    profile(
        "general_objects",
        "General objects",
        "detection",
        ["person", "car", "chair"],
        DOMAINS,
        "Object presence, not threat or intent",
        ["Bounding boxes and exhaustive object taxonomy"],
    ),
    profile(
        "person_detection",
        "People",
        "detection",
        ["person"],
        DOMAINS,
        "Person detection without identity recognition",
        ["Person boxes, occlusion and negative examples"],
    ),
    profile(
        "fire_smoke",
        "Fire and smoke",
        "detection",
        ["fire", "smoke"],
        DOMAINS,
        "Visible flame or smoke candidate for verification",
        [
            "Separate fire/smoke boxes",
            "Steam, clouds, glare and lighting negatives",
            "Independent source scenes",
        ],
    ),
    profile(
        "firearm",
        "Firearms",
        "detection",
        ["firearm"],
        DOMAINS,
        "Visible firearm-like object; does not determine intent",
        [
            "Firearm boxes and explicit subtype mapping",
            "Tools, replicas and other hard negatives",
            "Small-object and occlusion evaluation",
        ],
    ),
    profile(
        "fighting",
        "Fighting",
        "event",
        ["normal", "fighting"],
        DOMAINS,
        "Possible physical altercation requiring review",
        [
            "Temporally annotated clips",
            "Play, sport and ordinary interaction negatives",
            "Persistent entities or explicitly labeled fixed-duration scene clips",
        ],
    ),
    profile(
        "fall",
        "Falls",
        "event",
        ["normal", "fall"],
        DOMAINS,
        "Possible fall; not medical diagnosis",
        [
            "Fall onset and recovery clips",
            "Sitting, kneeling and lying-down negatives",
            "Subject/session-disjoint evaluation",
        ],
    ),
    profile(
        "theft",
        "Possible theft events",
        "event",
        ["normal", "possible_theft"],
        ["retail", "office", "warehouse", "parking"],
        "Review candidate, not a finding of theft or criminal intent",
        [
            "Reviewed temporal actions and operational context",
            "Ordinary handling and authorized removal negatives",
            "Human decision before incident confirmation",
        ],
    ),
    profile(
        "traffic_collision",
        "Traffic collisions",
        "event",
        ["normal", "collision"],
        ["traffic", "parking", "industrial"],
        "Possible collision requiring temporal verification",
        [
            "Collision and near-miss clips",
            "Vehicle tracks and independent locations",
            "Night/weather evaluation",
        ],
    ),
    profile(
        "traffic_violation",
        "Traffic rule review",
        "policy",
        ["car", "truck", "bus", "motorcycle", "bicycle"],
        ["traffic", "parking"],
        "Configured-rule observation, not an automatic legal finding",
        [
            "Per-site lane/direction/stop-line geometry",
            "Signal state and timestamps where relevant",
            "Metric calibration for speed; no speed claim from raw pixels",
        ],
    ),
    profile(
        "ppe_objects",
        "Protective equipment",
        "detection",
        ["person", "helmet", "safety_vest", "gloves"],
        ["industrial", "warehouse"],
        "Visible equipment; presence alone does not prove correct use",
        [
            "Equipment and body-part boxes",
            "Explicit dataset label mapping",
            "Small-object visibility evaluation",
        ],
    ),
    profile(
        "ppe_compliance",
        "PPE compliance review",
        "policy",
        ["person", "helmet", "safety_vest"],
        ["industrial", "warehouse"],
        "Possible missing equipment only when the relevant body region is observable",
        [
            "Site-specific required PPE",
            "Person-equipment association and body visibility",
            "Abstain on occlusion; persistence before review",
        ],
    ),
    profile(
        "industrial_hazards",
        "Industrial hazard review",
        "policy",
        ["person", "forklift"],
        ["industrial", "warehouse"],
        "Explicit configured hazards; no universal industrial-hazard detector",
        [
            "Separate restricted-zone, forklift-proximity, blocked-exit and machine-state policies",
            "Calibrated site geometry and equipment context",
            "Hazard-specific event labels and near-miss evaluation",
        ],
    ),
]
BY_ID = {p["id"]: p for p in PROFILES}
POLICY_PRIMITIVES = {
    "traffic_violation": ["wrong_way"],
    "ppe_compliance": ["possible_missing_helmet"],
    "industrial_hazards": ["restricted_zone"],
}


def validate_profiles(manifest):
    ids = manifest.get("capability_ids", [])
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("capability_ids must contain unique task profile IDs")
    for key in ids:
        if key not in BY_ID:
            raise ValueError(f"Unknown task profile: {key}")
        item = BY_ID[key]
        if item["task"] == "policy":
            raise ValueError("A detector dataset cannot establish a site-policy capability")
        if manifest.get("task", "event") != item["task"]:
            raise ValueError(f"Task does not match profile {key}")
        missing = set(item["required_classes"]) - set(manifest.get("classes", []))
        if missing:
            raise ValueError(f"Profile {key} requires explicitly mapped classes: {sorted(missing)}")
    return ids


def declared_profiles(manifest):
    # Backward compatibility for the earlier, explicitly person-only experiment.
    if (
        "capability_ids" not in manifest
        and manifest.get("domain") == "pedestrian"
        and manifest.get("classes") == ["person"]
    ):
        return ["person_detection"]
    return validate_profiles(manifest)


def coverage(models, datasets, active_model_ids=(), verified_model_ids=()):
    result = []
    for item in PROFILES:
        candidates = []
        for model in models:
            metadata = model["manifest"]
            try:
                matches = item["id"] in declared_profiles(metadata)
            except ValueError:
                matches = False
            if matches:
                candidates.append(
                    {
                        "id": model["id"],
                        "version": model["version"],
                        "stage": model["stage"],
                        "classes": metadata.get("classes", []),
                        "domain": metadata.get("domain"),
                        "synthetic": metadata.get("synthetic", False),
                        "calibrated": metadata.get("calibrated", False),
                        "origin": metadata.get("training_status", "locally_trained"),
                        "acceptance_approved": model["id"] in verified_model_ids,
                        "active": model["id"] in active_model_ids,
                    }
                )
        compatible = []
        for name, metadata in datasets:
            try:
                if item["id"] in declared_profiles(metadata):
                    compatible.append(name)
            except ValueError:
                pass
        result.append(
            {
                **item,
                "implemented_rules": POLICY_PRIMITIVES.get(item["id"], []),
                "datasets": compatible,
                "models": candidates,
                "status": "candidate_available"
                if candidates
                else "data_available"
                if compatible
                else "not_trained"
                if item["task"] != "policy"
                else "policy_partial"
                if item["id"] in POLICY_PRIMITIVES
                else "policy_not_implemented",
                "production_validated": any(
                    m["acceptance_approved"] and not m["synthetic"] and m["stage"] == "production"
                    for m in candidates
                ),
            }
        )
    return result
