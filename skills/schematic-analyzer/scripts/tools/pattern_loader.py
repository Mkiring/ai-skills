"""Pattern loader for subsystem detection.

Loads pattern definitions only from explicit runtime sources.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re
import yaml


@dataclass
class SignalPattern:
    """Signal pattern definition."""

    role: str
    patterns: list[str]
    required: bool = True
    group_key: bool = False


@dataclass
class Pattern:
    """Subsystem pattern definition."""

    name: str
    category: str
    description: str = ""
    signals: list[SignalPattern] = field(default_factory=list)
    controller: dict[str, Any] = field(default_factory=dict)
    participants: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def get_required_signals(self) -> list[SignalPattern]:
        """Get list of required signals."""
        return [s for s in self.signals if s.required]

    def get_group_key_signals(self) -> list[SignalPattern]:
        """Get list of signals used for grouping."""
        return [s for s in self.signals if s.group_key]

    def compile_patterns(self) -> dict[str, list[re.Pattern]]:
        """Compile regex patterns for each signal role.

        Returns:
            Dict mapping role -> list of compiled regex patterns
        """
        compiled = {}
        for signal in self.signals:
            compiled[signal.role] = [
                re.compile(p) for p in signal.patterns
            ]
        return compiled


class PatternLoader:
    """Load and manage subsystem patterns."""

    def __init__(
        self,
        project_path: Optional[str] = None,
        extra_sources: Optional[list[str] | tuple[str, ...]] = None,
    ):
        """Initialize pattern loader.

        Args:
            project_path: Optional project path for project-level patterns
            extra_sources: Optional runtime pattern YAML files or directories
        """
        self.project_path = Path(project_path) if project_path else None
        self.extra_sources = tuple(str(Path(source).resolve()) for source in extra_sources or ())
        self._pattern_cache: dict[str, Pattern] = {}

    def load(self, force_reload: bool = False) -> list[Pattern]:
        """Load all active patterns.

        Args:
            force_reload: Force reload from disk

        Returns:
            List of Pattern objects
        """
        if self._pattern_cache and not force_reload:
            return list(self._pattern_cache.values())

        patterns: dict[str, Pattern] = {}

        if self.extra_sources:
            for source in self.extra_sources:
                source_path = Path(source)
                if source_path.is_dir():
                    for pattern in self._load_dir(source_path):
                        patterns[pattern.name] = pattern
                elif source_path.is_file():
                    pattern = self._load_file(source_path)
                    if pattern:
                        patterns[pattern.name] = pattern
        self._pattern_cache = patterns
        return list(patterns.values())

    def get_pattern(self, name: str) -> Optional[Pattern]:
        """Get a specific pattern by name.

        Args:
            name: Pattern name (e.g., "I2C", "SPI")

        Returns:
            Pattern object or None
        """
        if not self._pattern_cache:
            self.load()
        return self._pattern_cache.get(name)

    def _load_dir(self, dir_path: Path) -> list[Pattern]:
        """Load all patterns from a directory.

        Args:
            dir_path: Directory containing .yaml pattern files

        Returns:
            List of loaded Pattern objects
        """
        patterns = []
        for yaml_file in dir_path.glob("*.yaml"):
            pattern = self._load_file(yaml_file)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _load_file(self, file_path: Path) -> Optional[Pattern]:
        """Load a pattern from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Pattern object or None on error
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                return None

            # Parse signals
            signals = []
            for sig_data in data.get("signals", []):
                signals.append(SignalPattern(
                    role=sig_data.get("role", ""),
                    patterns=sig_data.get("patterns", []),
                    required=sig_data.get("required", True),
                    group_key=sig_data.get("group_key", False),
                ))

            return Pattern(
                name=data.get("name", file_path.stem),
                category=data.get("category", "bus"),
                description=data.get("description", ""),
                signals=signals,
                controller=data.get("controller", {}),
                participants=data.get("participants", {}),
                extra_fields=data.get("extra_fields", {}),
            )

        except Exception as e:
            print(f"Warning: Failed to load pattern {file_path}: {e}", file=__import__("sys").stderr)
            return None
