"""Provisional output scorer for SAM Phase 1.

This is a stand-in for the real Phase 2 deterministic validator (spec'd in
BIM_Modeller_Build_Spec_Part_1.md, not yet built). It exists so Phase 1 base
model selection / fine-tune checks have *some* automated signal on whether a
generated floor plan is structurally sane. It intentionally does not attempt
the fuller geometric/adjacency/constraint checks the real Phase 2 validator
will do (room-to-room adjacency correctness, door/circulation reachability,
zoning rules, etc.) — do not treat a pass here as "the plan is good," only as
"the plan is not obviously broken."

Replace this module's caller with the real Phase 2 validator once it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Polygon

REQUIRED_TOP_LEVEL_KEYS = {"rooms", "adjacency", "total_area"}
REQUIRED_ROOM_KEYS = {"name", "type", "zone", "area", "polygon", "floor"}

MIN_ROOM_AREA_SQM = 0.01
MAX_ROOM_AREA_SQM = 130.0

# Below this, a partial overlap is treated as edge/shared-wall noise, not a defect.
OVERLAP_AREA_TOLERANCE_SQM = 0.5
# Above this fraction of the smaller room's area, the overlap is treated as an
# intentional nested/open-plan space (kitchen open to living, storage under a
# stair, etc.) rather than a double-booked area — this dataset uses that
# convention routinely, confirmed against sam/data/processed/comparison_subset_500.jsonl.
CONTAINED_OVERLAP_RATIO = 0.9
BOUNDS_TOLERANCE_M = 0.1

FRONT_DOOR_PREFIX = "front_door_"


def score_output(raw_json_text: str, site_width: float, site_length: float) -> dict:
    """Run provisional structural checks against one generated floor plan.

    Returns {'valid': bool, 'reasons': list[str]}. 'reasons' is empty when
    valid is True, otherwise contains one message per failed check.
    """
    reasons: list[str] = []

    try:
        data = json.loads(raw_json_text)
    except json.JSONDecodeError as exc:
        return {"valid": False, "reasons": [f"not valid JSON: {exc}"]}

    if not isinstance(data, dict) or set(data.keys()) != REQUIRED_TOP_LEVEL_KEYS:
        got = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
        reasons.append(
            f"top-level object must have exactly keys {sorted(REQUIRED_TOP_LEVEL_KEYS)}, got {got}"
        )
        return {"valid": False, "reasons": reasons}

    rooms = data["rooms"]
    adjacency = data["adjacency"]

    if not isinstance(rooms, list) or not rooms:
        return {"valid": False, "reasons": ["'rooms' must be a non-empty list"]}

    for i, room in enumerate(rooms):
        if not isinstance(room, dict) or set(room.keys()) != REQUIRED_ROOM_KEYS:
            got = sorted(room.keys()) if isinstance(room, dict) else type(room).__name__
            reasons.append(
                f"room[{i}] must have exactly keys {sorted(REQUIRED_ROOM_KEYS)}, got {got}"
            )

    if reasons:
        return {"valid": False, "reasons": reasons}

    for room in rooms:
        area = room["area"]
        if not isinstance(area, (int, float)) or not (
            MIN_ROOM_AREA_SQM <= area <= MAX_ROOM_AREA_SQM
        ):
            reasons.append(
                f"room '{room['name']}' area {area} outside allowed range "
                f"[{MIN_ROOM_AREA_SQM}, {MAX_ROOM_AREA_SQM}] sqm"
            )

    polygons: dict[str, Polygon] = {}
    for room in rooms:
        try:
            polygons[room["name"]] = Polygon(room["polygon"])
        except Exception as exc:
            reasons.append(f"room '{room['name']}' has an invalid polygon: {exc}")

    names = list(polygons.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = polygons[names[i]], polygons[names[j]]
            if not a.is_valid or not b.is_valid:
                continue
            intersection_area = a.intersection(b).area
            if intersection_area <= OVERLAP_AREA_TOLERANCE_SQM:
                continue
            smaller_area = min(a.area, b.area)
            if smaller_area > 0 and intersection_area > CONTAINED_OVERLAP_RATIO * smaller_area:
                # One room is (near-)fully nested in the other — treated as an
                # intentional open-plan/nested space, not a double-booked area.
                continue
            reasons.append(
                f"rooms '{names[i]}' and '{names[j]}' overlap by "
                f"{intersection_area:.3f} sqm"
            )

    # Room polygons aren't guaranteed to be anchored at the origin — the site
    # rectangle can be translated in the plan's coordinate space (confirmed
    # against real data: e.g. site.length=9.21 with rooms spanning y=3.83..12.57,
    # a span of 8.74m that fits fine, just not anchored at y=0). So check that
    # the building's overall footprint fits within site_width x site_length,
    # not that it starts at (0, 0).
    if polygons:
        all_bounds = [poly.bounds for poly in polygons.values()]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        footprint_width = max_x - min_x
        footprint_length = max_y - min_y
        if footprint_width > site_width + BOUNDS_TOLERANCE_M:
            reasons.append(
                f"building footprint width {footprint_width:.3f} exceeds "
                f"site_width {site_width}"
            )
        if footprint_length > site_length + BOUNDS_TOLERANCE_M:
            reasons.append(
                f"building footprint length {footprint_length:.3f} exceeds "
                f"site_length {site_length}"
            )

    room_names = set(polygons.keys())
    if not isinstance(adjacency, list):
        reasons.append("'adjacency' must be a list")
    else:
        for entry in adjacency:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                reasons.append(f"adjacency entry {entry!r} must be a 2-element list")
                continue
            for ref in entry:
                if ref in room_names:
                    continue
                if isinstance(ref, str) and ref.startswith(FRONT_DOOR_PREFIX):
                    continue
                reasons.append(
                    f"adjacency references unknown room '{ref}' "
                    f"(not in rooms and not a front_door_* entry)"
                )

    return {"valid": len(reasons) == 0, "reasons": reasons}


def score_file(path) -> dict:
    """Run score_output over every record in a .jsonl file shaped like
    sam/data/processed/*.jsonl: each line is {"messages": [system, user,
    assistant]}, where the user message's content is JSON containing
    {"site": {"width": ..., "length": ...}, ...} and the assistant message's
    content is the raw floor-plan JSON string to score.

    Returns an overall pass rate plus per-record results.
    """
    path = Path(path)
    results: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = {m["role"]: m["content"] for m in record["messages"]}
            user_payload = json.loads(messages["user"])
            site_width = user_payload["site"]["width"]
            site_length = user_payload["site"]["length"]
            raw_json_text = messages["assistant"]
            result = score_output(raw_json_text, site_width, site_length)
            result["line"] = line_no
            results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["valid"])
    return {
        "pass_rate": passed / total if total else 0.0,
        "total": total,
        "passed": passed,
        "results": results,
    }
