"""Core component detection for schematic analysis."""

from dataclasses import dataclass, field
from typing import Any

from .component_rules import (
    evaluate_role_matches,
    get_net_names,
    infer_transport_hints,
    load_rule_set,
    normalize_rule_paths,
    reference_matches_family,
)


@dataclass
class CoreComponent:
    """A core component identified in the schematic."""

    instance_id: str
    sheet_instance_path: str
    reference: str
    value: str
    lib_id: str
    role: str
    score: int
    priority: int
    pin_count: int
    net_count: int
    evidence: list[str] = field(default_factory=list)


def _score_from_bands(value: int, config_bands) -> int:
    for band in config_bands:
        if value >= band.minimum:
            return band.points
    return 0


def calculate_component_score(component: dict[str, Any], rule_paths: tuple[str, ...] = ()) -> int:
    """Calculate structural importance score for a component.

    Core ranking must stay structural so downstream role/subsystem inference does
    not leak back into core selection. Only pin count and net count contribute.
    """
    score = 0
    net_names = get_net_names(component)
    pin_count = len(component.get("pins", []))
    net_count = len(set(net_names))

    normalized_rule_paths = normalize_rule_paths(rule_paths)
    scoring = load_rule_set(normalized_rule_paths).core_scoring

    score += _score_from_bands(pin_count, scoring.pin_count_scores)
    score += _score_from_bands(net_count, scoring.net_count_scores)
    return score


def infer_role(
    value: str,
    lib_id: str,
    component: dict[str, Any] | None = None,
    rule_paths: tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """Infer component role from shared rule-based evidence."""
    candidate = dict(component or {})
    candidate.setdefault("value", value)
    candidate.setdefault("lib_id", lib_id)
    candidate.setdefault("pins", [])
    candidate.setdefault("nets", [])

    normalized_rule_paths = normalize_rule_paths(rule_paths)
    role_matches = evaluate_role_matches(candidate, normalized_rule_paths)
    if role_matches:
        best = role_matches[0]
        return best.role, list(best.evidence)

    hints = infer_transport_hints(candidate, normalized_rule_paths)
    if hints:
        return "Unknown", hints

    return "Unknown", []


def detect_core_components(
    components: list[dict[str, Any]],
    top_n: int = 5,
    rule_paths: tuple[str, ...] = (),
) -> list[CoreComponent]:
    """Identify core components in a schematic."""
    normalized_rule_paths = normalize_rule_paths(rule_paths)
    scoring = load_rule_set(normalized_rule_paths).core_scoring
    scored_components = []

    for component in components:
        reference = str(component.get("reference", "")).upper()
        if not reference:
            continue

        if any(
            reference_matches_family(reference, family, normalized_rule_paths)
            for family in scoring.excluded_reference_families
        ):
            continue

        score = calculate_component_score(component, normalized_rule_paths)
        if score < scoring.minimum_score:
            continue

        role, evidence = infer_role(
            str(component.get("value", "")),
            str(component.get("lib_id", "")),
            component,
            normalized_rule_paths,
        )
        sheet_instance_path = _normalize_sheet_instance_path(component.get("sheet_instance_path"))
        instance_id = str(component.get("instance_id") or _build_instance_id(reference, sheet_instance_path))
        scored_components.append(
            CoreComponent(
                instance_id=instance_id,
                sheet_instance_path=sheet_instance_path,
                reference=reference,
                value=str(component.get("value", "")),
                lib_id=str(component.get("lib_id", "")),
                role=role,
                score=score,
                priority=0,
                pin_count=len(component.get("pins", [])),
                net_count=len(set(get_net_names(component))),
                evidence=evidence,
            )
        )

    scored_components.sort(
        key=lambda component: (
            component.score,
            component.net_count,
            component.pin_count,
            component.reference,
        ),
        reverse=True,
    )

    for priority, component in enumerate(scored_components[:top_n], start=1):
        component.priority = priority

    return scored_components[:top_n]


def _normalize_sheet_instance_path(sheet_instance_path: Any) -> str:
    normalized = str(sheet_instance_path or "").strip().rstrip("/")
    return normalized or "/"


def _build_instance_id(reference: str, sheet_instance_path: str) -> str:
    ref = str(reference or "").upper()
    if sheet_instance_path in {"", "/"}:
        return f"/{ref}"
    return f"{sheet_instance_path}/{ref}"


def get_main_controller(core_components: list[CoreComponent]) -> CoreComponent | None:
    """Get the main controller (highest priority processor)."""
    for comp in core_components:
        if comp.role.startswith("Processor"):
            return comp
    return None


def get_power_ics(core_components: list[CoreComponent]) -> list[CoreComponent]:
    """Get all power-related core components."""
    return [c for c in core_components if c.role.startswith("Power")]
