"""Generic subsystem detector.

Detects subsystems using pattern definitions loaded from YAML files.
This replaces the hardcoded topology_extractor with a data-driven approach.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import re
from collections import defaultdict

# Ensure scripts directory is in path for imports
_SCRIPTS_DIR = Path(__file__).parent.parent.resolve()
if str(_SCRIPTS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPTS_DIR))

from tools.pattern_loader import Pattern, PatternLoader


@dataclass
class Subsystem:
    """Generic subsystem structure."""

    name: str
    type: str
    category: str

    # Signal mapping: role -> net_name
    signals: dict[str, str] = field(default_factory=dict)

    # Controller (if any)
    controller: dict[str, Any] | None = None

    # Participant devices
    participants: list[dict[str, Any]] = field(default_factory=list)

    # Subsystem-specific fields
    extra: dict[str, Any] = field(default_factory=dict)

    # Confidence score
    confidence: float = 1.0


class SubsystemDetector:
    """Detect subsystems from schematic data using patterns."""

    def __init__(
        self,
        project_path: Optional[str] = None,
        extra_pattern_sources: Optional[list[str] | tuple[str, ...]] = None,
    ):
        """Initialize detector.

        Args:
            project_path: Optional project path for custom patterns
            extra_pattern_sources: Runtime pattern YAML files or directories
        """
        self.loader = PatternLoader(project_path, extra_sources=extra_pattern_sources)
        self._patterns: list[Pattern] = []
        self._compiled_patterns: dict[str, dict[str, list[re.Pattern]]] = {}

    def load_patterns(self, force_reload: bool = False) -> list[Pattern]:
        """Load patterns from loader.

        Args:
            force_reload: Force reload from disk

        Returns:
            List of Pattern objects
        """
        if not self._patterns or force_reload:
            self._patterns = self.loader.load(force_reload)
            self._compiled_patterns = {}
            for p in self._patterns:
                self._compiled_patterns[p.name] = p.compile_patterns()
        return self._patterns

    def detect(
        self,
        components: list[dict[str, Any]],
        nets: list[dict[str, Any]],
        core_ref: Optional[str] = None,
    ) -> list[Subsystem]:
        """Detect all subsystems from schematic data.

        Args:
            components: List of component dicts from schematic
            nets: List of net dicts from schematic
            core_ref: Reference of main controller (for controller detection)

        Returns:
            List of detected Subsystem objects
        """
        patterns = self.load_patterns()
        subsystems = []

        for pattern in patterns:
            detected = self._detect_pattern(pattern, components, nets, core_ref)
            subsystems.extend(detected)

        return subsystems

    def _detect_pattern(
        self,
        pattern: Pattern,
        components: list[dict[str, Any]],
        nets: list[dict[str, Any]],
        core_ref: Optional[str],
    ) -> list[Subsystem]:
        """Detect subsystems matching a single pattern.

        Args:
            pattern: Pattern definition
            components: Component list
            nets: Net list
            core_ref: Main controller reference

        Returns:
            List of detected Subsystems for this pattern
        """
        compiled = self._compiled_patterns.get(pattern.name, {})

        # Step 1: Match nets to signal roles
        signal_matches = self._match_signals(nets, pattern, compiled)

        # Step 2: Group signals by group_key
        signal_groups = self._group_signals(signal_matches, pattern)

        # Step 3: For each valid group, find participants and build subsystem
        subsystems = []
        group_key_signals = pattern.get_group_key_signals()

        for group_id, group_signals in signal_groups.items():
            # Check if all required signals are present
            if not self._check_required_signals(group_signals, pattern):
                continue

            # Find participants connected to this signal group
            participants = self._find_participants(
                components, group_signals, pattern, core_ref
            )

            # Generic bus-name matches without any real attached devices are not
            # useful subsystem detections and tend to collapse distinct buses.
            if pattern.category == "bus" and not participants:
                continue

            # Identify controller
            controller = self._identify_controller(
                components, group_signals, pattern, core_ref
            )

            # Build subsystem name
            subsystem_name = f"{pattern.name}_{len(subsystems) + 1}"

            # Build signal dict
            signals_dict = {role: net_name for role, net_name in group_signals.items()}

            # Build extra fields (placeholder for now)
            extra = self._build_extra_fields(pattern, participants)

            subsystem = Subsystem(
                name=subsystem_name,
                type=pattern.name,
                category=pattern.category,
                signals=signals_dict,
                controller=controller,
                participants=participants,
                extra=extra,
                confidence=self._calculate_confidence(group_signals, participants, pattern),
            )
            subsystems.append(subsystem)

        return subsystems

    def _match_signals(
        self,
        nets: list[dict[str, Any]],
        pattern: Pattern,
        compiled: dict[str, list[re.Pattern]],
    ) -> dict[str, list[tuple[str, str]]]:
        """Match nets to signal roles.

        Args:
            nets: List of net dicts
            pattern: Pattern definition
            compiled: Compiled regex patterns

        Returns:
            Dict mapping role -> list of (net_name, net_code) tuples
        """
        matches: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for net in nets:
            net_name = net.get("name", "")
            net_code = net.get("code", "")

            for signal in pattern.signals:
                regexes = compiled.get(signal.role, [])
                for regex in regexes:
                    if regex.search(net_name):
                        matches[signal.role].append((net_name, net_code))
                        break

        return dict(matches)

    def _group_signals(
        self,
        signal_matches: dict[str, list[tuple[str, str]]],
        pattern: Pattern,
    ) -> dict[str, dict[str, str]]:
        """Group matched signals by their group_key suffix.

        Args:
            signal_matches: Dict of role -> [(net_name, net_code), ...]
            pattern: Pattern definition

        Returns:
            Dict mapping group_id -> {role: net_name}
        """
        group_key_signals = pattern.get_group_key_signals()
        if not group_key_signals:
            # No group keys, treat all matches as one group
            result = {}
            all_matched = True
            group_signals = {}

            for signal in pattern.signals:
                if signal.role in signal_matches and signal_matches[signal.role]:
                    group_signals[signal.role] = signal_matches[signal.role][0][0]
                elif signal.required:
                    all_matched = False

            if all_matched and group_signals:
                result["default"] = group_signals
            return result

        # Extract group key suffix from first group_key signal
        first_group_signal = group_key_signals[0]
        if first_group_signal.role not in signal_matches:
            return {}

        groups: dict[str, dict[str, str]] = defaultdict(dict)

        # Use first group key signal to determine groups
        for net_name, net_code in signal_matches.get(first_group_signal.role, []):
            # Extract suffix (everything after the signal pattern match)
            group_id = self._extract_group_id(net_name, first_group_signal)

            # Try to find matching signals for other roles
            group_signals = {first_group_signal.role: net_name}
            valid_group = True

            for signal in pattern.signals:
                if signal.role == first_group_signal.role:
                    continue

                # Find matching net for this role in the same group
                matched = self._find_matching_signal(
                    signal_matches.get(signal.role, []),
                    group_id,
                    signal,
                )

                if matched:
                    group_signals[signal.role] = matched
                elif signal.required:
                    valid_group = False
                    break

            if valid_group:
                groups[group_id] = dict(group_signals)

        return dict(groups)

    def _extract_group_id(self, net_name: str, signal_pattern) -> str:
        """Extract group identifier from net name.

        Args:
            net_name: Net name (e.g., "I2C1_SDA")
            signal_pattern: SignalPattern object

        Returns:
            Group identifier (e.g., "I2C1")
        """
        # Try to extract common suffix patterns
        # E.g., I2C1_SDA -> I2C1, SPI_MOSI_1 -> 1, etc.
        for pattern in signal_pattern.patterns:
            match = re.search(pattern, net_name)
            if match:
                # Get the part before the match or use the whole prefix
                # Common pattern: PREFIX_SUFFIX -> PREFIX
                parts = net_name.split("_")
                if len(parts) >= 2:
                    # Try to find the numeric suffix
                    for i, part in enumerate(parts):
                        if re.match(r"\d+", part):
                            return "_".join(parts[:i + 1])
                    # No numeric, use first part
                    return parts[0]
                return net_name

        return net_name

    def _find_matching_signal(
        self,
        candidates: list[tuple[str, str]],
        group_id: str,
        signal_pattern,
    ) -> Optional[str]:
        """Find a signal matching the group_id.

        Args:
            candidates: List of (net_name, net_code) candidates
            group_id: Group identifier to match
            signal_pattern: SignalPattern object

        Returns:
            Matching net name or None
        """
        for net_name, net_code in candidates:
            # Check if this net belongs to the same group
            net_group = self._extract_group_id(net_name, signal_pattern)
            if net_group == group_id:
                return net_name

        # Fallback: if only one candidate, use it
        if len(candidates) == 1:
            return candidates[0][0]

        return None

    def _check_required_signals(
        self,
        group_signals: dict[str, str],
        pattern: Pattern,
    ) -> bool:
        """Check if all required signals are present.

        Args:
            group_signals: Dict of role -> net_name
            pattern: Pattern definition

        Returns:
            True if all required signals present
        """
        for signal in pattern.get_required_signals():
            if signal.role not in group_signals:
                return False
        return True

    def _find_participants(
        self,
        components: list[dict[str, Any]],
        group_signals: dict[str, str],
        pattern: Pattern,
        core_ref: Optional[str],
    ) -> list[dict[str, Any]]:
        """Find components participating in this subsystem.

        Args:
            components: List of component dicts
            group_signals: Dict of role -> net_name for this group
            pattern: Pattern definition
            core_ref: Main controller reference

        Returns:
            List of participant dicts
        """
        participants = []
        participant_config = pattern.participants
        filter_config = participant_config.get("filter", [])
        exclude_controller = participant_config.get("exclude_controller", True)

        # Get all signal net names
        signal_nets = set(group_signals.values())

        for comp in components:
            ref = comp.get("reference", "")

            # Skip if excluded
            if exclude_controller and core_ref and ref == core_ref:
                continue

            # Check filter rules
            if self._matches_filter(ref, filter_config):
                continue

            # Check if component is connected to any signal in this group
            comp_nets = self._get_component_nets(comp)
            if not signal_nets.intersection(comp_nets):
                continue

            # Must be connected to at least one required signal
            required_nets = {
                group_signals[s.role]
                for s in pattern.get_required_signals()
                if s.role in group_signals
            }
            if not required_nets.intersection(comp_nets):
                continue

            participants.append({
                "ref": ref,
                "value": comp.get("value", ""),
                "role": self._determine_participant_role(comp, pattern),
                "extra": {},
            })

        return participants

    def _get_component_nets(self, comp: dict[str, Any]) -> set[str]:
        """Get set of net names connected to a component.

        Args:
            comp: Component dict

        Returns:
            Set of net names
        """
        nets = set()

        # Try component_nets if available (from Phase 1)
        if "connected_nets" in comp:
            nets.update(comp.get("connected_nets", []))

        # Try nets list
        for net in comp.get("nets", []):
            if isinstance(net, dict):
                nets.add(net.get("name", ""))
            elif isinstance(net, str):
                nets.add(net)

        return nets

    def _matches_filter(self, ref: str, filter_config: list[dict]) -> bool:
        """Check if component reference matches filter rules.

        Args:
            ref: Component reference (e.g., "R10", "U15")
            filter_config: List of filter rules

        Returns:
            True if component should be filtered out
        """
        for rule in filter_config:
            if "reference_prefix" in rule:
                prefixes = rule["reference_prefix"]
                if any(ref.startswith(p) for p in prefixes):
                    return True
        return False

    def _determine_participant_role(self, comp: dict[str, Any], pattern: Pattern) -> str:
        """Determine the role of a participant in the subsystem.

        Args:
            comp: Component dict
            pattern: Pattern definition

        Returns:
            Role string (e.g., "device", "converter", "transceiver")
        """
        # Check component role from classification
        role = comp.get("role", {})
        if isinstance(role, dict):
            primary = role.get("primary", "")
            if primary:
                return primary.lower().replace("/", "_")

        # Default based on category
        if pattern.category == "bus":
            return "device"
        elif pattern.category == "tree":
            return "node"
        else:
            return "participant"

    def _identify_controller(
        self,
        components: list[dict[str, Any]],
        group_signals: dict[str, str],
        pattern: Pattern,
        core_ref: Optional[str],
    ) -> dict[str, Any] | None:
        """Identify the controller for this subsystem.

        Args:
            components: Component list
            group_signals: Signal dict for this group
            pattern: Pattern definition
            core_ref: Main controller reference

        Returns:
            Controller dict or None
        """
        controller_config = pattern.controller
        detect_by = controller_config.get("detect_by", "none")

        if detect_by == "none":
            return None

        if detect_by == "core_component" and core_ref:
            return {"ref": core_ref}

        if detect_by == "pin_direction":
            # Would need pin direction info from netlist
            # For now, fall back to core_component
            if core_ref:
                return {"ref": core_ref}

        return None

    def _build_extra_fields(
        self,
        pattern: Pattern,
        participants: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build extra fields for subsystem.

        Args:
            pattern: Pattern definition
            participants: List of participants

        Returns:
            Dict of extra fields
        """
        extra = {}

        for field_name, field_config in pattern.extra_fields.items():
            source = field_config.get("source", "static")

            if source == "static":
                extra[field_name] = field_config.get("default")
            elif source == "infer":
                # Placeholder - would need more analysis or datasheet lookup
                extra[field_name] = None

        return extra

    def _calculate_confidence(
        self,
        group_signals: dict[str, str],
        participants: list[dict[str, Any]],
        pattern: Pattern,
    ) -> float:
        """Calculate confidence score for this detection.

        Args:
            group_signals: Matched signals
            participants: Found participants
            pattern: Pattern definition

        Returns:
            Confidence score (0.0 - 1.0)
        """
        score = 0.5  # Base score

        # More required signals matched = higher confidence
        required_count = len(pattern.get_required_signals())
        matched_required = len([
            s for s in pattern.get_required_signals()
            if s.role in group_signals
        ])
        if required_count > 0:
            score += 0.3 * (matched_required / required_count)

        # More participants = higher confidence
        if len(participants) >= 2:
            score += 0.1
        if len(participants) >= 4:
            score += 0.1

        return min(score, 1.0)


def to_dict(subsystem: Subsystem) -> dict[str, Any]:
    """Convert Subsystem to dictionary for JSON output.

    Args:
        subsystem: Subsystem object

    Returns:
        Dict representation
    """
    return {
        "name": subsystem.name,
        "type": subsystem.type,
        "category": subsystem.category,
        "signals": subsystem.signals,
        "controller": subsystem.controller,
        "participants": subsystem.participants,
        "extra": subsystem.extra,
        "confidence": subsystem.confidence,
    }
