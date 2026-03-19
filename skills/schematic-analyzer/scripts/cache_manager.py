"""Cache management for schematic analysis.

Provides caching with:
- File hash-based invalidation
- Incremental updates
- Error detection and correction
"""

import hashlib
import json
import random
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheMeta:
    """Metadata for cached analysis."""

    schematic_path: str
    schematic_hash: str
    context_signature: str
    analyzed_at: str
    cache_key: str = ""
    version: str = "1.0"
    phases_completed: list[int] = field(default_factory=list)
    errors_detected: list[dict[str, Any]] = field(default_factory=list)


class CacheManager:
    """Manages analysis cache with validation and correction."""

    CACHE_DIR_NAME = ".schematic-analyzer"
    CACHE_VERSION = "1.6"

    def __init__(self, project_path: str):
        """Initialize cache manager.

        Args:
            project_path: Path to the KiCad project directory
        """
        self.project_path = Path(project_path)
        self.cache_root = self.project_path / self.CACHE_DIR_NAME / "cache"
        self.live_root = self.cache_root
        self.tags_root = self.cache_root / "tags"

    def _prune_legacy_hash_dirs(self) -> None:
        """Remove pre-live-slot hash directories left by older cache layouts."""
        if not self.cache_root.exists():
            return
        for child in self.cache_root.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"analysis", "structural", "tags"}:
                continue
            if len(child.name) == 16 and all(ch in "0123456789abcdef" for ch in child.name.lower()):
                shutil.rmtree(child)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex hash string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # Use first 16 chars as hash

    def compute_schematic_hash(self, schematic_path: str | Path) -> str:
        """Compute the raw hash of a schematic file."""
        return self._compute_file_hash(Path(schematic_path))

    def compute_cache_key(self, schematic_path: str | Path, context_signature: str = "") -> str:
        """Compute cache key from schematic content and runtime context."""
        schematic_hash = self.compute_schematic_hash(schematic_path)
        if not context_signature:
            return schematic_hash

        sha256 = hashlib.sha256()
        sha256.update(schematic_hash.encode("utf-8"))
        sha256.update(b":")
        sha256.update(context_signature.encode("utf-8"))
        return sha256.hexdigest()[:16]

    def _analysis_dir(self) -> Path:
        """Return the fixed live slot for final analysis cache."""
        return self.live_root / "analysis"

    def _structural_dir(self) -> Path:
        """Return the fixed live slot for structural cache."""
        return self.live_root / "structural"

    def _tags_dir(self) -> Path:
        """Return the root directory for user-managed cache tags."""
        return self.tags_root

    def _phase_dir(self, phase: int) -> Path:
        """Return the live slot directory for the requested phase."""
        return self._structural_dir() if phase in {0, 1} else self._analysis_dir()

    def _load_meta(self, cache_dir: Path) -> Optional[CacheMeta]:
        """Load cache metadata.

        Args:
            cache_dir: Path to cache directory

        Returns:
            CacheMeta or None if not found
        """
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CacheMeta(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _save_meta(self, cache_dir: Path, meta: CacheMeta) -> None:
        """Save cache metadata.

        Args:
            cache_dir: Path to cache directory
            meta: Metadata to save
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

    def should_refresh(
        self,
        schematic_path: str,
        force: bool = False,
        context_signature: str = "",
    ) -> tuple[bool, str, Optional[str]]:
        """Check if cache needs refresh.

        Args:
            schematic_path: Path to schematic file
            force: Force refresh regardless of cache state

        Returns:
            (need_refresh, reason, cached_hash)
        """
        if force:
            return True, "forced", None

        schematic = Path(schematic_path)
        if not schematic.exists():
            return True, "schematic_not_found", None

        # Compute current hash
        current_hash = self.compute_schematic_hash(schematic)
        cache_key = self.compute_cache_key(schematic, context_signature)
        cache_dir = self._analysis_dir()

        # Check if cache exists
        if not cache_dir.exists():
            return True, "no_cache", None

        # Load and validate meta
        meta = self._load_meta(cache_dir)
        if not meta:
            return True, "cache_corrupted", cache_key

        # Check hash match
        if meta.cache_key != cache_key:
            return True, "runtime_context_changed", meta.cache_key or cache_key

        if meta.schematic_hash != current_hash:
            return True, "schematic_changed", cache_key

        if meta.context_signature != context_signature:
            return True, "runtime_context_changed", cache_key

        if meta.version != self.CACHE_VERSION:
            return True, "cache_version_mismatch", cache_key

        # Check cache integrity
        analysis_path = cache_dir / "analysis.json"
        if not analysis_path.exists():
            return True, "cache_incomplete", cache_key

        # Validate data consistency
        if not self._validate_data_consistency(schematic, cache_dir):
            return True, "data_inconsistent", cache_key

        return False, "cache_valid", cache_key

    def _validate_data_consistency(
        self,
        schematic_path: Path,
        cache_dir: Path,
    ) -> bool:
        """Validate cached data against schematic.

        Uses sampling to check consistency without full re-parse.

        Args:
            schematic_path: Path to schematic file
            cache_dir: Path to cache directory

        Returns:
            True if consistent, False otherwise
        """
        analysis_path = cache_dir / "analysis.json"
        if not analysis_path.exists():
            return False

        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        # Sample validation: check a few components exist
        components = cached.get("components", [])
        if not components:
            return True  # Empty is valid

        # Sample 5 components
        sample_size = min(5, len(components))
        samples = random.sample(components, sample_size)

        # Read the root schematic and sibling sheets so hierarchical references
        # are validated against the full project, not just the root page.
        contents = []
        for candidate in sorted(self.project_path.glob("*.kicad_sch")):
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                    contents.append(f.read())
            except IOError:
                return False

        if not contents:
            return False

        for comp in samples:
            ref = comp.get("reference", "")
            if not ref:
                continue

            if not any(ref in content for content in contents):
                return False

        return True

    def get_cached_analysis(self, schematic_hash: str) -> Optional[dict[str, Any]]:
        """Load cached analysis data.

        Args:
            schematic_hash: Hash of the schematic file

        Returns:
            Analysis dict or None if not found
        """
        cache_dir = self._analysis_dir()
        meta = self._load_meta(cache_dir)
        if not meta or meta.cache_key != schematic_hash:
            return None
        analysis_path = cache_dir / "analysis.json"

        if not analysis_path.exists():
            return None

        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def save_analysis(
        self,
        schematic_path: str,
        analysis: dict[str, Any],
        phases_completed: list[int],
        context_signature: str = "",
    ) -> str:
        """Save analysis to cache.

        Args:
            schematic_path: Path to schematic file
            analysis: Analysis data dict
            phases_completed: List of completed phase numbers

        Returns:
            Cache hash
        """
        schematic = Path(schematic_path)
        current_hash = self.compute_schematic_hash(schematic)
        cache_key = self.compute_cache_key(schematic, context_signature)
        self._prune_legacy_hash_dirs()
        cache_dir = self._analysis_dir()

        # Ensure directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save analysis
        analysis_path = cache_dir / "analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Save metadata
        meta = CacheMeta(
            schematic_path=str(schematic_path),
            schematic_hash=current_hash,
            cache_key=cache_key,
            context_signature=context_signature,
            analyzed_at=datetime.now().isoformat(),
            version=self.CACHE_VERSION,
            phases_completed=phases_completed,
        )
        self._save_meta(cache_dir, meta)

        return cache_key

    def save_phase_result(
        self,
        schematic_hash: str,
        phase: int,
        data: dict[str, Any],
    ) -> None:
        """Save intermediate phase result.

        Args:
            schematic_hash: Hash of the schematic file
            phase: Phase number (0, 1, 2, 3)
            data: Phase result data
        """
        self._prune_legacy_hash_dirs()
        cache_dir = self._phase_dir(phase)
        cache_dir.mkdir(parents=True, exist_ok=True)

        meta = self._load_meta(cache_dir)
        context_signature = ""
        if phase in {2, 3}:
            context_signature = schematic_hash
        meta = CacheMeta(
            schematic_path=meta.schematic_path if meta else "",
            schematic_hash=meta.schematic_hash if meta else "",
            cache_key=schematic_hash,
            context_signature=context_signature,
            analyzed_at=datetime.now().isoformat(),
            version=self.CACHE_VERSION,
            phases_completed=sorted({*(meta.phases_completed if meta else []), phase}),
            errors_detected=list(meta.errors_detected) if meta else [],
        )
        self._save_meta(cache_dir, meta)

        phase_path = cache_dir / f"phase_{phase}.json"
        with open(phase_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_phase_result(
        self,
        schematic_hash: str,
        phase: int,
    ) -> Optional[dict[str, Any]]:
        """Load intermediate phase result.

        Args:
            schematic_hash: Hash of the schematic file
            phase: Phase number (0, 1, 2, 3)

        Returns:
            Phase data or None if not found
        """
        cache_dir = self._phase_dir(phase)
        meta = self._load_meta(cache_dir)
        if not meta or meta.cache_key != schematic_hash:
            return None
        phase_path = cache_dir / f"phase_{phase}.json"

        if not phase_path.exists():
            return None

        try:
            with open(phase_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def record_error(
        self,
        schematic_hash: str,
        error_type: str,
        error_details: str,
        location: Optional[str] = None,
    ) -> None:
        """Record a detected error in cache.

        Args:
            schematic_hash: Hash of the schematic file
            error_type: Type of error
            error_details: Detailed error description
            location: Optional location (component ref, net name, etc.)
        """
        cache_dir = self._analysis_dir() if error_type.startswith("analysis") else self._structural_dir()
        meta = self._load_meta(cache_dir)

        if meta:
            meta.errors_detected.append({
                "type": error_type,
                "details": error_details,
                "location": location,
                "timestamp": datetime.now().isoformat(),
            })
            self._save_meta(cache_dir, meta)

    def clear_cache(self, schematic_hash: Optional[str] = None) -> None:
        """Clear cache.

        Args:
            schematic_hash: Specific hash to clear, or None for all
        """
        if schematic_hash:
            for cache_dir in (self._analysis_dir(), self._structural_dir()):
                meta = self._load_meta(cache_dir)
                if meta and meta.cache_key == schematic_hash and cache_dir.exists():
                    shutil.rmtree(cache_dir)
            return

        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)

    def create_tag(self, tag: str) -> Path:
        """Create a user-managed snapshot of the current live cache."""
        normalized = tag.strip()
        if not normalized:
            raise ValueError("Tag name must not be empty")

        tag_dir = self._tags_dir() / normalized
        if tag_dir.exists():
            raise ValueError(f"Cache tag already exists: {normalized}")

        if not self._analysis_dir().exists() and not self._structural_dir().exists():
            raise RuntimeError("No live cache available to tag")

        tag_dir.parent.mkdir(parents=True, exist_ok=True)
        tag_dir.mkdir(parents=True, exist_ok=False)

        if self._analysis_dir().exists():
            shutil.copytree(self._analysis_dir(), tag_dir / "analysis")
        if self._structural_dir().exists():
            shutil.copytree(self._structural_dir(), tag_dir / "structural")

        return tag_dir

    def list_entries(self) -> list[dict[str, Any]]:
        """List live cache slots and user-managed tag snapshots."""
        entries: list[dict[str, Any]] = []

        if self._analysis_dir().exists() or self._structural_dir().exists():
            analysis_meta = self._load_meta(self._analysis_dir())
            structural_meta = self._load_meta(self._structural_dir())
            analyzed_at = ""
            if analysis_meta:
                analyzed_at = analysis_meta.analyzed_at
            elif structural_meta:
                analyzed_at = structural_meta.analyzed_at
            entries.append(
                {
                    "kind": "live",
                    "name": "current",
                    "path": self.live_root,
                    "analyzed_at": analyzed_at,
                    "analysis_cache_key": analysis_meta.cache_key if analysis_meta else "",
                    "structural_cache_key": structural_meta.cache_key if structural_meta else "",
                }
            )

        if self._tags_dir().exists():
            for tag_dir in sorted(path for path in self._tags_dir().iterdir() if path.is_dir()):
                analysis_meta = self._load_meta(tag_dir / "analysis")
                structural_meta = self._load_meta(tag_dir / "structural")
                analyzed_at = ""
                if analysis_meta:
                    analyzed_at = analysis_meta.analyzed_at
                elif structural_meta:
                    analyzed_at = structural_meta.analyzed_at
                entries.append(
                    {
                        "kind": "tag",
                        "name": tag_dir.name,
                        "path": tag_dir,
                        "analyzed_at": analyzed_at,
                        "analysis_cache_key": analysis_meta.cache_key if analysis_meta else "",
                        "structural_cache_key": structural_meta.cache_key if structural_meta else "",
                    }
                )

        return entries
