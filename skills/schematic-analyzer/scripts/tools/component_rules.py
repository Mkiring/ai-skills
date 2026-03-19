"""Shared, data-driven component role rules."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import re

import yaml


def get_builtin_rule_paths() -> tuple[str, ...]:
    """Return built-in rule sources shipped with the skill."""
    rules_dir = Path(__file__).resolve().parents[2] / "rules"
    candidate = rules_dir / "component_rules.yaml"
    if candidate.exists():
        return (str(candidate.resolve()),)
    return ()


def normalize_rule_paths(rule_paths: Iterable[str] | None = None) -> tuple[str, ...]:
    """Normalize runtime rule paths into a stable, absolute tuple."""
    normalized = []
    seen = set()
    for raw_path in rule_paths or ():
        resolved = str(Path(raw_path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return tuple(normalized)


@dataclass(frozen=True)
class RoleRule:
    """A role classification rule loaded from YAML."""

    role: str
    min_score: float
    core_score: int
    value_patterns: tuple[re.Pattern[str], ...]
    lib_patterns: tuple[re.Pattern[str], ...]
    pin_patterns: tuple[re.Pattern[str], ...]
    net_patterns: tuple[re.Pattern[str], ...]
    required_groups: tuple["RequiredSignalGroup", ...]


@dataclass(frozen=True)
class RequiredSignalGroup:
    """A signal group that must be present for a role to match."""

    name: str
    pin_patterns: tuple[re.Pattern[str], ...]
    net_patterns: tuple[re.Pattern[str], ...]
    min_matches: int
    score: float


@dataclass(frozen=True)
class TransportHint:
    """A bus or transport participation hint."""

    label: str
    pin_patterns: tuple[re.Pattern[str], ...]
    net_patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class RoleMatch:
    """A scored role match for a component."""

    role: str
    score: float
    confidence: float
    core_score: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RuleSet:
    """All component inference rules."""

    roles: tuple[RoleRule, ...]
    transport_hints: tuple[TransportHint, ...]
    reference_prefixes: dict[str, tuple[str, ...]]
    mcp_category_map: dict[str, str]
    mcp_subcategory_map: dict[str, str]
    core_scoring: "CoreScoringConfig"


@dataclass(frozen=True)
class ScoreBand:
    """A thresholded scoring band."""

    minimum: int
    points: int


@dataclass(frozen=True)
class TransportHintBonus:
    """Bonus awarded when transport hints are present."""

    minimum_pin_count: int
    points: int


@dataclass(frozen=True)
class CoreScoringConfig:
    """Configurable core-component scoring heuristics."""

    excluded_reference_families: tuple[str, ...]
    minimum_score: int
    signal_pin_patterns: tuple[re.Pattern[str], ...]
    pin_count_scores: tuple[ScoreBand, ...]
    net_count_scores: tuple[ScoreBand, ...]
    signal_pin_scores: tuple[ScoreBand, ...]
    transport_hint_bonus: TransportHintBonus


@dataclass(frozen=True)
class RuleSourceDescriptor:
    """A cache key descriptor for a rule source file."""

    path: str
    mtime_ns: int
    size: int


def _compile_patterns(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


def _read_yaml_doc(file_path: str) -> dict[str, Any]:
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_named_entries(
    base_entries: list[dict[str, Any]],
    overlay_entries: list[dict[str, Any]],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}

    for entry in base_entries:
        key = entry.get(key_name)
        if not key:
            continue
        index_by_key[key] = len(merged)
        merged.append(deepcopy(entry))

    for entry in overlay_entries:
        key = entry.get(key_name)
        if not key:
            continue
        if key in index_by_key:
            merged[index_by_key[key]] = _deep_merge_dict(merged[index_by_key[key]], entry)
        else:
            index_by_key[key] = len(merged)
            merged.append(deepcopy(entry))

    return merged


def _merge_reference_prefixes(
    base_prefixes: dict[str, list[str]],
    overlay_prefixes: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged = deepcopy(base_prefixes)
    for family, prefixes in overlay_prefixes.items():
        merged[family] = list(prefixes)
    return merged


def _merge_rule_docs(source_paths: tuple[str, ...]) -> dict[str, Any]:
    merged_doc: dict[str, Any] = {
        "roles": [],
        "transport_hints": [],
        "reference_prefixes": {},
        "mcp_category_map": {},
        "mcp_subcategory_map": {},
        "core_scoring": {},
    }

    for source_path in source_paths:
        doc = _read_yaml_doc(source_path)
        merged_doc["roles"] = _merge_named_entries(
            merged_doc.get("roles", []),
            doc.get("roles", []),
            key_name="role",
        )
        merged_doc["transport_hints"] = _merge_named_entries(
            merged_doc.get("transport_hints", []),
            doc.get("transport_hints", []),
            key_name="label",
        )
        merged_doc["reference_prefixes"] = _merge_reference_prefixes(
            merged_doc.get("reference_prefixes", {}),
            doc.get("reference_prefixes", {}),
        )
        merged_doc["mcp_category_map"] = {
            **merged_doc.get("mcp_category_map", {}),
            **doc.get("mcp_category_map", {}),
        }
        merged_doc["mcp_subcategory_map"] = {
            **merged_doc.get("mcp_subcategory_map", {}),
            **doc.get("mcp_subcategory_map", {}),
        }
        merged_doc["core_scoring"] = _deep_merge_dict(
            merged_doc.get("core_scoring", {}),
            doc.get("core_scoring", {}),
        )

    return merged_doc


def _describe_rule_sources(source_paths: tuple[str, ...]) -> tuple[RuleSourceDescriptor, ...]:
    descriptors = []
    for source_path in source_paths:
        stat = Path(source_path).stat()
        descriptors.append(
            RuleSourceDescriptor(
                path=source_path,
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
            )
        )
    return tuple(descriptors)


def _parse_score_bands(entries: list[dict[str, Any]]) -> tuple[ScoreBand, ...]:
    bands = [
        ScoreBand(
            minimum=int(entry.get("minimum", 0)),
            points=int(entry.get("points", 0)),
        )
        for entry in entries
    ]
    bands.sort(key=lambda band: band.minimum, reverse=True)
    return tuple(bands)


def _parse_required_groups(entries: list[dict[str, Any]]) -> tuple[RequiredSignalGroup, ...]:
    groups = []
    for entry in entries:
        groups.append(
            RequiredSignalGroup(
                name=str(entry.get("name", "")).strip(),
                pin_patterns=_compile_patterns(entry.get("pin_patterns", [])),
                net_patterns=_compile_patterns(entry.get("net_patterns", [])),
                min_matches=int(entry.get("min_matches", 1)),
                score=float(entry.get("score", 2.5)),
            )
        )
    return tuple(group for group in groups if group.name)


@lru_cache(maxsize=32)
def _load_rule_set_cached(source_descriptors: tuple[RuleSourceDescriptor, ...]) -> RuleSet:
    source_paths = tuple(descriptor.path for descriptor in source_descriptors)
    raw = _merge_rule_docs(source_paths)

    roles = []
    for role_data in raw.get("roles", []):
        roles.append(
            RoleRule(
                role=role_data["role"],
                min_score=float(role_data.get("min_score", 4.0)),
                core_score=int(role_data.get("core_score", 0)),
                value_patterns=_compile_patterns(role_data.get("value_patterns", [])),
                lib_patterns=_compile_patterns(role_data.get("lib_patterns", [])),
                pin_patterns=_compile_patterns(role_data.get("pin_patterns", [])),
                net_patterns=_compile_patterns(role_data.get("net_patterns", [])),
                required_groups=_parse_required_groups(role_data.get("required_groups", [])),
            )
        )

    transport_hints = []
    for hint_data in raw.get("transport_hints", []):
        transport_hints.append(
            TransportHint(
                label=hint_data["label"],
                pin_patterns=_compile_patterns(hint_data.get("pin_patterns", [])),
                net_patterns=_compile_patterns(hint_data.get("net_patterns", [])),
            )
        )

    reference_prefixes = {
        family: tuple(prefixes)
        for family, prefixes in raw.get("reference_prefixes", {}).items()
    }

    core_scoring_raw = raw.get("core_scoring", {})
    core_scoring = CoreScoringConfig(
        excluded_reference_families=tuple(core_scoring_raw.get("excluded_reference_families", [])),
        minimum_score=int(core_scoring_raw.get("minimum_score", 10)),
        signal_pin_patterns=_compile_patterns(core_scoring_raw.get("signal_pin_patterns", [])),
        pin_count_scores=_parse_score_bands(core_scoring_raw.get("pin_count_scores", [])),
        net_count_scores=_parse_score_bands(core_scoring_raw.get("net_count_scores", [])),
        signal_pin_scores=_parse_score_bands(core_scoring_raw.get("signal_pin_scores", [])),
        transport_hint_bonus=TransportHintBonus(
            minimum_pin_count=int(core_scoring_raw.get("transport_hint_bonus", {}).get("minimum_pin_count", 8)),
            points=int(core_scoring_raw.get("transport_hint_bonus", {}).get("points", 3)),
        ),
    )

    return RuleSet(
        roles=tuple(roles),
        transport_hints=tuple(transport_hints),
        reference_prefixes=reference_prefixes,
        mcp_category_map=dict(raw.get("mcp_category_map", {})),
        mcp_subcategory_map=dict(raw.get("mcp_subcategory_map", {})),
        core_scoring=core_scoring,
    )


def load_rule_set(extra_rule_paths: tuple[str, ...] = ()) -> RuleSet:
    """Load component rules from built-in and runtime sources."""
    source_paths = get_builtin_rule_paths() + normalize_rule_paths(extra_rule_paths)
    return _load_rule_set_cached(_describe_rule_sources(source_paths))


def get_pin_names(component: dict[str, Any]) -> list[str]:
    """Extract normalized pin names from a component dict."""
    pin_names = []
    for pin in component.get("pins", []):
        if isinstance(pin, dict):
            name = str(pin.get("name", "")).strip()
        else:
            name = str(getattr(pin, "name", "")).strip()
        if name:
            pin_names.append(name)
    return pin_names


def get_net_names(component: dict[str, Any]) -> list[str]:
    """Extract normalized net names from a component dict."""
    return [str(net).strip() for net in component.get("nets", []) if str(net).strip()]


def reference_matches_family(reference: str, family: str, rule_paths: tuple[str, ...] = ()) -> bool:
    """Check whether a reference designator belongs to a configured family."""
    prefixes = load_rule_set(normalize_rule_paths(rule_paths)).reference_prefixes.get(family, ())
    return reference.upper().startswith(tuple(prefix.upper() for prefix in prefixes))


def _match_patterns(texts: list[str], patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    """Return unique texts matched by any pattern in the rule."""
    matches = []
    seen = set()
    for text in texts:
        for pattern in patterns:
            if pattern.search(text):
                if text not in seen:
                    matches.append(text)
                    seen.add(text)
                break
    return matches


def _score_match_count(count: int, *, base: float, step: float, cap: float) -> float:
    if count <= 0:
        return 0.0
    return min(cap, base + max(0, count - 1) * step)


def _confidence_from_score(score: float) -> float:
    if score >= 9.0:
        return 0.95
    if score >= 7.0:
        return 0.88
    if score >= 5.5:
        return 0.8
    if score >= 4.0:
        return 0.7
    return 0.0


def evaluate_role_matches(component: dict[str, Any], rule_paths: tuple[str, ...] = ()) -> list[RoleMatch]:
    """Evaluate all configured role rules for a component."""
    rules = load_rule_set(normalize_rule_paths(rule_paths))

    value = str(component.get("value", "")).strip()
    lib_id = str(component.get("lib_id", "")).strip()
    pin_names = get_pin_names(component)
    net_names = get_net_names(component)

    matches: list[RoleMatch] = []

    for rule in rules.roles:
        value_matches = _match_patterns([value], rule.value_patterns)
        lib_matches = _match_patterns([lib_id], rule.lib_patterns)
        pin_matches = _match_patterns(pin_names, rule.pin_patterns)
        net_matches = _match_patterns(net_names, rule.net_patterns)

        score = 0.0
        evidence = []

        value_score = _score_match_count(len(value_matches), base=4.0, step=0.5, cap=5.0)
        if value_score:
            score += value_score
            evidence.append(f"value={', '.join(value_matches[:2])}")

        lib_score = _score_match_count(len(lib_matches), base=4.0, step=0.5, cap=5.0)
        if lib_score:
            score += lib_score
            evidence.append(f"lib={', '.join(lib_matches[:2])}")

        pin_score = _score_match_count(len(pin_matches), base=1.5, step=0.75, cap=3.5)
        if pin_score:
            score += pin_score
            evidence.append(f"pins={', '.join(pin_matches[:3])}")

        net_score = _score_match_count(len(net_matches), base=1.0, step=0.5, cap=2.5)
        if net_score:
            score += net_score
            evidence.append(f"nets={', '.join(net_matches[:3])}")

        required_groups_satisfied = True
        for group in rule.required_groups:
            group_pin_matches = _match_patterns(pin_names, group.pin_patterns)
            group_net_matches = _match_patterns(net_names, group.net_patterns)
            group_match_count = len(group_pin_matches) + len(group_net_matches)
            if group_match_count < group.min_matches:
                required_groups_satisfied = False
                break

            score += group.score
            group_evidence = list(group_pin_matches[:2]) + list(group_net_matches[:2])
            evidence.append(f"group:{group.name}={', '.join(group_evidence[:3])}")

        if not required_groups_satisfied:
            continue

        if score < rule.min_score:
            continue

        matches.append(
            RoleMatch(
                role=rule.role,
                score=score,
                confidence=_confidence_from_score(score),
                core_score=rule.core_score,
                evidence=tuple(evidence),
            )
        )

    matches.sort(key=lambda match: (match.score, match.core_score, match.role), reverse=True)
    return matches


def infer_transport_hints(component: dict[str, Any], rule_paths: tuple[str, ...] = ()) -> list[str]:
    """Infer transport/bus participation hints from pins and nets."""
    rules = load_rule_set(normalize_rule_paths(rule_paths))
    pin_names = get_pin_names(component)
    net_names = get_net_names(component)

    hints = []
    for hint in rules.transport_hints:
        pin_matches = _match_patterns(pin_names, hint.pin_patterns)
        net_matches = _match_patterns(net_names, hint.net_patterns)
        if pin_matches or net_matches:
            hints.append(hint.label)

    return hints
