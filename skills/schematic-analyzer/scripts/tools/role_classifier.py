"""Component role classification for schematic analysis."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .component_rules import (
    evaluate_role_matches,
    infer_transport_hints,
    load_rule_set,
    normalize_rule_paths,
    reference_matches_family,
)


@dataclass
class ComponentRole:
    """Role classification for a component."""

    reference: str
    primary: str
    secondary: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = "unknown"


def _map_text_to_role(text: str, mapping: dict[str, str]) -> str:
    """Map an external category string to a configured role."""
    if text in mapping:
        return mapping[text]

    text_lower = text.lower()
    for candidate, role in mapping.items():
        candidate_lower = candidate.lower()
        if candidate_lower in text_lower or text_lower in candidate_lower:
            return role

    return "Unknown"


def map_mcp_category(category: str, subcategory: str = "", rule_paths: tuple[str, ...] = ()) -> str:
    """Map MCP category to schematic role."""
    normalized_rule_paths = normalize_rule_paths(rule_paths)
    rules = load_rule_set(normalized_rule_paths)

    category_role = _map_text_to_role(category, rules.mcp_category_map)
    if category_role != "Unknown":
        return category_role

    return _map_text_to_role(subcategory, rules.mcp_subcategory_map)


def _role_from_reference(reference: str, rule_paths: tuple[str, ...] = ()) -> Optional[ComponentRole]:
    """Infer trivial roles from reference designators."""
    normalized_rule_paths = normalize_rule_paths(rule_paths)
    if reference_matches_family(reference, "Passive", normalized_rule_paths):
        return ComponentRole(reference=reference, primary="Passive", confidence=1.0, evidence=["reference=passive"], source="reference")
    if reference_matches_family(reference, "Connector", normalized_rule_paths):
        return ComponentRole(reference=reference, primary="Connector", confidence=0.95, evidence=["reference=connector"], source="reference")
    return None


def _transport_secondary(primary: str, hints: list[str]) -> list[str]:
    """Convert transport hints into secondary roles."""
    secondary = []
    for hint in hints:
        label = hint
        if primary.startswith("Processor"):
            label = hint.replace("Participant", "Controller")
        if label not in secondary:
            secondary.append(label)
    return secondary


def classify_component(
    component: dict[str, Any],
    mcp_info: Optional[dict[str, Any]] = None,
    rule_paths: tuple[str, ...] = (),
) -> ComponentRole:
    """Classify a component's role using multiple evidence sources."""
    ref = component.get("reference", "")
    normalized_rule_paths = normalize_rule_paths(rule_paths)

    trivial_role = _role_from_reference(ref, normalized_rule_paths)
    if trivial_role:
        return trivial_role

    evidence: list[str] = []
    primary = "Unknown"
    confidence = 0.0
    source = "unknown"

    if mcp_info:
        category = mcp_info.get("category", "")
        subcategory = mcp_info.get("subcategory", "")
        if category:
            primary = map_mcp_category(category, subcategory, normalized_rule_paths)
            evidence.append(f"mcp={category}/{subcategory}")
            confidence = 0.95 if primary != "Unknown" else 0.7
            source = "mcp"

    if confidence < 0.9:
        matches = evaluate_role_matches(component, normalized_rule_paths)
        if matches:
            best = matches[0]
            if best.confidence > confidence:
                primary = best.role
                confidence = best.confidence
                evidence = list(best.evidence)
                source = "rule_engine"

    transport_hints = infer_transport_hints(component, normalized_rule_paths)
    secondary = _transport_secondary(primary, transport_hints)

    return ComponentRole(
        reference=ref,
        primary=primary,
        secondary=secondary,
        confidence=confidence,
        evidence=evidence,
        source=source,
    )
