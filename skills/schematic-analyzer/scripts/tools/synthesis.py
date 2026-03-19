"""Synthesis layer for interpreted schematic results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .overlay_interpreter import PatternMatch


@dataclass(frozen=True)
class SubsystemConclusion:
    """Final subsystem conclusion exposed to CLI and JSON outputs."""

    name: str
    type: str
    category: str
    signals: dict[str, str] = field(default_factory=dict)
    controller: dict[str, Any] | None = None
    participants: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_source: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a final conclusion into the established CLI shape."""
        return {
            "name": self.name,
            "type": self.type,
            "category": self.category,
            "signals": dict(self.signals),
            "controller": self.controller,
            "participants": list(self.participants),
            "extra": dict(self.extra),
            "confidence": self.confidence,
            "evidence_source": list(self.evidence_source),
        }


@dataclass(frozen=True)
class ArchitectureSummary:
    """Structured synthesis summary for future layered cache reuse."""

    subsystems: list[SubsystemConclusion] = field(default_factory=list)


@dataclass(frozen=True)
class SynthesisResult:
    """Complete synthesis output."""

    summary: ArchitectureSummary
    markdown_report: str


class SynthesisEngine:
    """Convert explicit interpreted overlays into final subsystem conclusions."""

    def __init__(
        self,
        *,
        root_schematic: Path | str,
        analysis_schematic_paths: list[Path] | tuple[Path, ...],
        local_parser_factory: Callable[[Path], Any],
    ) -> None:
        self.root_schematic = Path(root_schematic).resolve()
        self.analysis_schematic_paths = [Path(path).resolve() for path in analysis_schematic_paths]
        self.local_parser_factory = local_parser_factory

    def synthesize(
        self,
        *,
        core_components: list[dict[str, Any]],
        components: list[dict[str, Any]],
        hierarchy: list[dict[str, Any]],
        interpreted_subsystems: list[PatternMatch],
        main_controller: str | None,
    ) -> list[dict[str, Any]]:
        """Build final subsystem conclusions from explicit interpreted patterns only."""
        del core_components
        del components
        del hierarchy
        del main_controller
        return [
            self._pattern_match_to_conclusion(subsystem).to_dict()
            for subsystem in interpreted_subsystems
        ]

    def generate_markdown_report(self, result: Any) -> str:
        """Render the existing Markdown report format from final analysis output."""
        lines = [
            f"# {result.meta.get('project_name', 'Schematic')} Analysis Report",
            "",
            "## Overview",
            f"- **Schematic**: {result.meta.get('schematic_path', '')}",
            f"- **Analyzed**: {result.meta.get('analyzed_at', '')}",
            f"- **Total Components**: {result.statistics.get('total_components', 0)}",
            f"- **Total Nets**: {result.statistics.get('total_nets', 0)}",
            f"- **Subsystems Identified**: {result.statistics.get('subsystem_count', 0)}",
            "",
        ]

        if result.core_components:
            lines.extend(
                [
                    "## Core Components",
                    "",
                    "| Priority | Reference | Value | Role | Score |",
                    "|----------|-----------|-------|------|-------|",
                ]
            )
            for component in result.core_components:
                lines.append(
                    f"| {component.get('priority', 0)} | {component.get('reference', '')} | "
                    f"{component.get('value', '')} | {component.get('role', '')} | {component.get('score', 0)} |"
                )
            lines.append("")

        if result.subsystems:
            lines.extend(["## Subsystems", ""])

            for subsystem in result.subsystems:
                subsystem_type = subsystem.get("type", "Unknown")
                subsystem_name = subsystem.get("name", "")
                subsystem_category = subsystem.get("category", "")

                lines.append(f"### {subsystem_name} ({subsystem_type})")
                if subsystem_category:
                    lines.append(f"*Category: {subsystem_category}*")
                lines.append("")

                signals = subsystem.get("signals", {})
                if signals:
                    lines.append("**Signals:**")
                    for role, net_name in signals.items():
                        lines.append(f"- {role}: {net_name}")
                    lines.append("")

                controller = subsystem.get("controller")
                if controller:
                    lines.append(f"**Controller:** {controller.get('ref', '')}")
                    lines.append("")

                participants = subsystem.get("participants", [])
                if participants:
                    lines.append(f"**Participants ({len(participants)}):**")
                    for participant in participants:
                        extra_info = ""
                        if participant.get("extra"):
                            extra_parts = [f"{key}={value}" for key, value in participant.get("extra", {}).items() if value]
                            if extra_parts:
                                extra_info = f" [{', '.join(extra_parts)}]"
                        lines.append(f"- {participant.get('ref', '')} ({participant.get('value', '')}){extra_info}")
                    lines.append("")

                extra = subsystem.get("extra", {})
                if extra:
                    lines.append("**Details:**")
                    for key, value in extra.items():
                        if value is not None:
                            lines.append(f"- {key}: {value}")
                    lines.append("")

                confidence = subsystem.get("confidence", 0)
                if confidence > 0:
                    lines.append(f"*Confidence: {confidence:.0%}*")
                    lines.append("")

        return "\n".join(lines)

    def _pattern_match_to_conclusion(self, subsystem: PatternMatch) -> SubsystemConclusion:
        return SubsystemConclusion(
            name=subsystem.name,
            type=subsystem.pattern_name,
            category=subsystem.category,
            signals={role: signal.net_name for role, signal in subsystem.signals.items()},
            controller=subsystem.controller,
            participants=subsystem.participants,
            extra=dict(subsystem.extra),
            confidence=subsystem.confidence,
            evidence_source=[subsystem.pattern_name],
        )
