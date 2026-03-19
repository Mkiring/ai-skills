"""Facade for runtime rule and pattern interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .role_classifier import classify_component
from .subsystem_detector import SubsystemDetector, Subsystem


@dataclass(frozen=True)
class InterpretedRole:
    """Role interpretation for one component instance."""

    instance_id: str
    reference: str
    primary: str
    secondary: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = "unknown"


@dataclass(frozen=True)
class NetMatch:
    """Interpreted net match with identity metadata."""

    net_name: str
    net_id: str = ""
    identity_space: str = "root"
    sheet_instance_path: str | None = None


@dataclass
class PatternMatch:
    """Pattern-driven interpreted subsystem match."""

    pattern_name: str
    name: str
    category: str
    signals: dict[str, NetMatch]
    controller: dict[str, Any] | None
    participants: list[dict[str, Any]]
    instance_ids: list[str]
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlayInterpretation:
    """Combined overlay interpretation payload."""

    roles: dict[str, InterpretedRole]
    patterns: list[PatternMatch]


class OverlayInterpreter:
    """Thin facade over role and subsystem interpretation engines."""

    def __init__(
        self,
        project_path: Optional[str] = None,
        rule_paths: Optional[tuple[str, ...] | list[str]] = None,
        pattern_sources: Optional[tuple[str, ...] | list[str]] = None,
    ) -> None:
        self.project_path = str(Path(project_path).resolve()) if project_path else None
        self.rule_paths = tuple(str(Path(path).resolve()) for path in (rule_paths or ()))
        self.pattern_sources = tuple(str(Path(path).resolve()) for path in (pattern_sources or ()))
        self._detector = SubsystemDetector(
            self.project_path,
            extra_pattern_sources=self.pattern_sources,
        )

    def interpret_components(
        self,
        components: list[dict[str, Any]],
        *,
        mcp_info_by_instance: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, InterpretedRole]:
        """Interpret component roles keyed by instance_id."""
        roles: dict[str, InterpretedRole] = {}
        mcp_info_by_instance = mcp_info_by_instance or {}

        for component in components:
            instance_id = self._component_instance_id(component)
            if not instance_id:
                continue

            interpreted = classify_component(
                component,
                mcp_info=mcp_info_by_instance.get(instance_id),
                rule_paths=self.rule_paths,
            )
            roles[instance_id] = InterpretedRole(
                instance_id=instance_id,
                reference=component.get("reference", ""),
                primary=interpreted.primary,
                secondary=list(interpreted.secondary),
                confidence=interpreted.confidence,
                evidence=list(interpreted.evidence),
                source=interpreted.source,
            )

        return roles

    def interpret_subsystems(
        self,
        *,
        components: list[dict[str, Any]],
        nets: list[dict[str, Any]],
        core_ref: Optional[str] = None,
    ) -> list[PatternMatch]:
        """Interpret subsystem patterns and enrich them with instance metadata."""
        detected = self._detector.detect(components, nets, core_ref)
        return [self._to_pattern_match(match, components, nets) for match in detected]

    def loaded_pattern_names(self) -> list[str]:
        """Return the currently active pattern names after all overlay loading."""
        return [pattern.name for pattern in self._detector.load_patterns()]

    def interpret(
        self,
        *,
        components: list[dict[str, Any]],
        nets: list[dict[str, Any]],
        core_ref: Optional[str] = None,
        mcp_info_by_instance: Optional[dict[str, dict[str, Any]]] = None,
    ) -> OverlayInterpretation:
        """Interpret both roles and pattern matches in one call."""
        return OverlayInterpretation(
            roles=self.interpret_components(components, mcp_info_by_instance=mcp_info_by_instance),
            patterns=self.interpret_subsystems(components=components, nets=nets, core_ref=core_ref),
        )

    def _to_pattern_match(
        self,
        subsystem: Subsystem,
        components: list[dict[str, Any]],
        nets: list[dict[str, Any]],
    ) -> PatternMatch:
        net_lookup = {str(net.get("name", "")): net for net in nets if net.get("name")}
        signal_net_names = {net_name for net_name in subsystem.signals.values() if net_name}

        signals = {
            role: NetMatch(
                net_name=net_name,
                net_id=str(net_lookup.get(net_name, {}).get("net_id", "")),
                identity_space=str(net_lookup.get(net_name, {}).get("identity_space", "root")),
                sheet_instance_path=net_lookup.get(net_name, {}).get("sheet_instance_path"),
            )
            for role, net_name in subsystem.signals.items()
        }

        controller = self._enrich_ref_dict(subsystem.controller, components, signal_net_names)
        participants = [
            participant
            for raw_participant in subsystem.participants
            for participant in self._expand_participant(raw_participant, components, signal_net_names)
        ]

        instance_ids: list[str] = []
        if controller and controller.get("instance_id"):
            instance_ids.append(controller["instance_id"])
        for participant in participants:
            instance_id = participant.get("instance_id")
            if instance_id and instance_id not in instance_ids:
                instance_ids.append(instance_id)

        return PatternMatch(
            pattern_name=subsystem.type,
            name=subsystem.name,
            category=subsystem.category,
            signals=signals,
            controller=controller,
            participants=participants,
            instance_ids=instance_ids,
            confidence=subsystem.confidence,
            extra=dict(subsystem.extra),
        )

    def _expand_participant(
        self,
        participant: dict[str, Any],
        components: list[dict[str, Any]],
        signal_net_names: set[str],
    ) -> list[dict[str, Any]]:
        ref = str(participant.get("ref", ""))
        value = str(participant.get("value", ""))
        matches = self._find_component_matches(ref, value, components, signal_net_names)
        if not matches:
            return [dict(participant)]

        enriched: list[dict[str, Any]] = []
        for component in matches:
            entry = dict(participant)
            entry["instance_id"] = self._component_instance_id(component)
            enriched.append(entry)
        return enriched

    def _enrich_ref_dict(
        self,
        raw: dict[str, Any] | None,
        components: list[dict[str, Any]],
        signal_net_names: set[str],
    ) -> dict[str, Any] | None:
        if not raw:
            return None

        enriched = dict(raw)
        ref = str(raw.get("ref", ""))
        matches = self._find_component_matches(ref, "", components, signal_net_names)
        if matches:
            enriched["instance_id"] = self._component_instance_id(matches[0])
        return enriched

    def _find_component_matches(
        self,
        ref: str,
        value: str,
        components: list[dict[str, Any]],
        signal_net_names: set[str],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for component in components:
            if str(component.get("reference", "")) != ref:
                continue
            if value and str(component.get("value", "")) != value:
                continue
            component_nets = {str(net) for net in component.get("nets", []) if str(net)}
            if signal_net_names and component_nets and not component_nets.intersection(signal_net_names):
                continue
            matches.append(component)
        return matches

    def _component_instance_id(self, component: dict[str, Any]) -> str:
        instance_id = str(component.get("instance_id", "")).strip()
        if instance_id:
            return instance_id

        ref = str(component.get("reference", "")).upper().strip()
        sheet_instance_path = str(component.get("sheet_instance_path", "/") or "/").rstrip("/") or "/"
        if not ref:
            return ""
        if sheet_instance_path in {"", "/"}:
            return f"/{ref}"
        return f"{sheet_instance_path}/{ref}"
