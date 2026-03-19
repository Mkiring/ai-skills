"""Project indexing for ERC-clean hierarchical schematics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .kicad.schematic_parser import SchematicParser
from .scope_resolver import ProjectScope, ScopeResolver


class UnknownReferenceError(KeyError):
    """Raised when a reference cannot be resolved."""


@dataclass(frozen=True)
class ComponentInstance:
    """One unique component in the project hierarchy."""

    reference: str
    instance_id: str
    value: str
    lib_id: str
    footprint: Optional[str]
    source_schematic: Path
    sheet_path: str
    sheet_type: str
    pins: list[dict]
    properties: dict[str, str]
    flags: dict[str, bool]

    @property
    def sheet_instance_path(self) -> str:
        return self.sheet_path


@dataclass(frozen=True)
class SheetInfo:
    """Metadata for one page in the hierarchy."""

    sheet_name: str
    sheet_file: str
    sheet_path: str
    sheet_type: str
    component_count: int

    @property
    def sheet_instance_path(self) -> str:
        return self.sheet_path


@dataclass(frozen=True)
class IndexStatistics:
    """Summary counts for the project index."""

    total_components: int
    total_sheets: int
    duplicate_reference_count: int


@dataclass
class InstanceResolver:
    """Resolve user-facing references in an ERC-clean project."""

    ref_to_instances: dict[str, list[str]]

    def resolve_ref(
        self,
        ref: str,
        sheet_name: Optional[str] = None,
        sheet_name_to_path: Optional[dict[str, str]] = None,
    ) -> str:
        del sheet_name, sheet_name_to_path
        ref_key = str(ref).upper()
        matches = list(self.ref_to_instances.get(ref_key, ()))
        if not matches:
            raise UnknownReferenceError(ref)
        return matches[0]


@dataclass(frozen=True)
class NetInfo:
    """One unique project net."""

    net_name: str
    net_code: int | None
    net_type: str


@dataclass(frozen=True)
class ProjectIndex:
    """Indexed project structure keyed by reference and net name."""

    scope: ProjectScope
    components: dict[str, ComponentInstance]
    nets: dict[str, NetInfo]
    hierarchy: list[SheetInfo]
    resolver: InstanceResolver
    sheet_name_to_path: dict[str, str]
    statistics: IndexStatistics


class ProjectIndexer:
    """Build a structural project index from the resolved hierarchy only."""

    def build(
        self,
        path: str | Path,
    ) -> ProjectIndex:
        input_path = Path(path).resolve()
        scope = ScopeResolver.resolve(input_path)

        components: dict[str, ComponentInstance] = {}
        hierarchy: list[SheetInfo] = []
        sheet_name_to_path: dict[str, str] = {}

        for record in self._build_sheet_records(scope):
            parser = SchematicParser(str(record.file_path), include_child_sheets=False)
            local_components = parser.get_components()
            hierarchy.append(
                SheetInfo(
                    sheet_name=record.sheet_name,
                    sheet_file=record.file_path.name,
                    sheet_path=record.sheet_path,
                    sheet_type=record.sheet_type,
                    component_count=len(local_components),
                )
            )
            sheet_name_to_path[record.sheet_name] = record.sheet_path

            for component in local_components:
                reference = component.reference.upper()
                pins = []
                for pin in component.pins:
                    if isinstance(pin, dict):
                        pins.append({"number": pin.get("number", ""), "name": pin.get("name", "")})
                    else:
                        pins.append(
                            {
                                "number": getattr(pin, "number", ""),
                                "name": getattr(pin, "name", ""),
                            }
                        )

                components[reference] = ComponentInstance(
                    reference=reference,
                    instance_id=reference,
                    value=component.value,
                    lib_id=component.library_id,
                    footprint=component.footprint,
                    source_schematic=record.file_path,
                    sheet_path=record.sheet_path,
                    sheet_type=record.sheet_type,
                    pins=pins,
                    properties=dict(component.properties),
                    flags=dict(component.flags),
                )

        return ProjectIndex(
            scope=scope,
            components=components,
            nets={},
            hierarchy=hierarchy,
            resolver=InstanceResolver({ref: [ref] for ref in sorted(components)}),
            sheet_name_to_path=sheet_name_to_path,
            statistics=IndexStatistics(
                total_components=len(components),
                total_sheets=len(hierarchy),
                duplicate_reference_count=0,
            ),
        )

    def _build_sheet_records(self, scope: ProjectScope) -> list["_SheetRecord"]:
        records: list[_SheetRecord] = [
            _SheetRecord(
                sheet_name=scope.project_name,
                file_path=scope.root_schematic,
                sheet_path="/",
                sheet_type="root",
            )
        ]
        root_parser = SchematicParser(str(scope.root_schematic), include_child_sheets=False)
        for sheet in root_parser.get_sheet_instance_records():
            source_schematic = Path(sheet["source_schematic"]).resolve()
            child_file = (source_schematic.parent / sheet["sheet_file"]).resolve()
            if not child_file.exists():
                continue
            records.append(
                _SheetRecord(
                    sheet_name=sheet["sheet_name"],
                    file_path=child_file,
                    sheet_path=sheet["sheet_name_path"],
                    sheet_type="hierarchy",
                )
            )
        return records


@dataclass(frozen=True)
class _SheetRecord:
    """Internal sheet traversal record."""

    sheet_name: str
    file_path: Path
    sheet_path: str
    sheet_type: str
