"""Structural evidence cards for schematic analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .connectivity_builder import ConnectivityGraph
from .core_ranker import CoreCandidate
from .kicad.schematic_parser import SchematicParser
from .project_indexer import ProjectIndex, SheetInfo


@dataclass
class ComponentCard:
    """Structural evidence for one component instance."""

    instance_id: str
    reference: str
    value: str
    sheet: str
    pin_count: int
    net_count: int
    connected_rails: list[str]
    top_neighbors: list[str]
    structural_rank: int


@dataclass
class SheetCard:
    """Structural evidence for one sheet."""

    sheet_name: str
    sheet_file: str
    component_count: int
    active_components: list[str]
    port_labels: list[str]
    local_nets: list[str]


@dataclass
class NeighborhoodCard:
    """1-hop/2-hop structural neighborhood around a component."""

    center: str
    hop_1: list[str]
    hop_2: list[str]
    shared_nets: dict[str, list[str]]
    limitation: Optional[str] = None


@dataclass
class RailSummary:
    """Structural summary for a power rail."""

    rail_name: str
    load_count: int
    loads: list[str]
    source: Optional[str] = None


class EvidenceExtractor:
    """Extract structural evidence cards from index + connectivity."""

    def __init__(
        self,
        project_index: ProjectIndex,
        connectivity_graph: ConnectivityGraph,
        components: list[dict],
        core_candidates: list[CoreCandidate],
        root_schematic: str | Path,
    ) -> None:
        self.project_index = project_index
        self.connectivity_graph = connectivity_graph
        self.components = list(components)
        self.components_by_instance = {
            component.get("instance_id", ""): component
            for component in self.components
            if component.get("instance_id")
        }
        self.rank_by_instance = {
            candidate.instance_id: candidate.priority
            for candidate in core_candidates
        }
        self.core_candidates = list(core_candidates)
        self.root_schematic = Path(root_schematic).resolve()
        self.sheet_info_by_path = {
            sheet.sheet_instance_path: sheet
            for sheet in project_index.hierarchy
        }
        self.sheet_path_to_name = {
            "/": "Root",
            **{
                sheet_path: sheet_name
                for sheet_name, sheet_path in project_index.sheet_name_to_path.items()
            },
        }

    def extract_top_core_cards(self, top_n: int = 5, max_items: int = 10) -> list[ComponentCard]:
        """Return default top-N component cards for LLM entry."""
        cards: list[ComponentCard] = []
        for candidate in self.core_candidates[:top_n]:
            cards.append(self.extract_component_card(candidate.instance_id, max_items=max_items))
        return cards

    def extract_component_card(self, instance_id: str, max_items: int = 10) -> ComponentCard:
        """Return a structural component card."""
        component = self._require_component(instance_id)
        neighbor_counts, _shared_nets = self._shared_neighbor_counts(instance_id)
        connected_rails = sorted(
            net.net_name
            for net in self.connectivity_graph.root_nets.values()
            if net.net_type == "Power" and instance_id in net.connected_instances
        )
        top_neighbors = [
            neighbor
            for neighbor, _count in neighbor_counts.most_common(max_items)
        ]

        return ComponentCard(
            instance_id=instance_id,
            reference=component.get("reference", ""),
            value=component.get("value", ""),
            sheet=self.sheet_path_to_name.get(component.get("sheet_instance_path", "/"), "Root"),
            pin_count=len(component.get("pins", [])),
            net_count=len(component.get("nets", [])),
            connected_rails=connected_rails[:max_items],
            top_neighbors=top_neighbors,
            structural_rank=self.rank_by_instance.get(instance_id, 0),
        )

    def extract_sheet_card(self, sheet_instance_path: str, max_items: int = 50) -> SheetCard:
        """Return a structural sheet card."""
        sheet_info = self._require_sheet(sheet_instance_path)
        parser = SchematicParser(str(self._sheet_file_path(sheet_info)), include_child_sheets=False)
        local_nets = parser.get_nets()
        sorted_local_nets = sorted(
            local_nets,
            key=lambda net: (
                len(self.connectivity_graph.root_nets.get(getattr(net, "name", ""), []).connected_instances)
                if getattr(net, "name", "") in self.connectivity_graph.root_nets
                else 0,
                getattr(net, "name", ""),
            ),
            reverse=True,
        )

        active_components = [
            component.get("instance_id", "")
            for component in self.components
            if component.get("sheet_instance_path") == sheet_instance_path
            and not self._is_passive(component.get("reference", ""))
        ]
        port_labels = []
        for net in local_nets:
            if getattr(net, "type", "") in {"global", "hierarchical"}:
                net_name = getattr(net, "name", "")
                if net_name and net_name not in port_labels:
                    port_labels.append(net_name)

        return SheetCard(
            sheet_name=sheet_info.sheet_name,
            sheet_file=sheet_info.sheet_file,
            component_count=len(
                [component for component in self.components if component.get("sheet_instance_path") == sheet_instance_path]
            ),
            active_components=active_components[:max_items],
            port_labels=port_labels[:max_items],
            local_nets=[getattr(net, "name", "") for net in sorted_local_nets[:max_items] if getattr(net, "name", "")],
        )

    def extract_neighborhood_card(
        self,
        instance_id: str,
        *,
        depth: int = 1,
        max_items: int = 50,
    ) -> NeighborhoodCard:
        """Return a structural neighborhood card."""
        component = self._require_component(instance_id)
        if instance_id not in self.connectivity_graph.component_nets:
            limitation = "sheet_local_only" if component.get("sheet_type") == "supplemental" else "pin_connectivity_unavailable"
            return NeighborhoodCard(
                center=instance_id,
                hop_1=[],
                hop_2=[],
                shared_nets={},
                limitation=limitation,
            )

        neighbor_counts, shared_nets = self._shared_neighbor_counts(instance_id)
        hop_1 = [neighbor for neighbor, _count in neighbor_counts.most_common(max_items)]
        hop_2: list[str] = []
        if depth >= 2:
            second_hop_counts: Counter[str] = Counter()
            hop_1_set = set(hop_1)
            for neighbor in hop_1:
                nested_counts, _nested_shared = self._shared_neighbor_counts(neighbor)
                for nested_neighbor, count in nested_counts.items():
                    if nested_neighbor == instance_id or nested_neighbor in hop_1_set:
                        continue
                    second_hop_counts[nested_neighbor] += count
            hop_2 = [neighbor for neighbor, _count in second_hop_counts.most_common(max_items)]

        truncated_shared_nets: dict[str, list[str]] = {}
        for net_name, peers in shared_nets.items():
            truncated_shared_nets[net_name] = peers[:max_items]

        return NeighborhoodCard(
            center=instance_id,
            hop_1=hop_1,
            hop_2=hop_2,
            shared_nets=truncated_shared_nets,
            limitation=None,
        )

    def extract_rail_summary(self, rail_name: str, max_items: int = 50) -> RailSummary:
        """Return structural load summary for a root-space rail."""
        connection = self.connectivity_graph.root_nets.get(rail_name)
        if connection is None or connection.net_type != "Power":
            raise KeyError(rail_name)

        loads = list(connection.connected_instances)
        return RailSummary(
            rail_name=rail_name,
            load_count=len(loads),
            loads=loads[:max_items],
            source=None,
        )

    def _shared_neighbor_counts(self, instance_id: str) -> tuple[Counter[str], dict[str, list[str]]]:
        neighbor_counts: Counter[str] = Counter()
        shared_nets: dict[str, list[str]] = {}

        for net in self.connectivity_graph.root_nets.values():
            if instance_id not in net.connected_instances:
                continue
            peers = [
                peer
                for peer in net.connected_instances
                if peer != instance_id
            ]
            if not peers:
                continue
            shared_nets[net.net_name] = peers
            for peer in peers:
                neighbor_counts[peer] += 1

        return neighbor_counts, shared_nets

    def _require_component(self, instance_id: str) -> dict:
        component = self.components_by_instance.get(instance_id)
        if component is None:
            raise KeyError(instance_id)
        return component

    def _require_sheet(self, sheet_instance_path: str) -> SheetInfo:
        sheet = self.sheet_info_by_path.get(sheet_instance_path)
        if sheet is None:
            raise KeyError(sheet_instance_path)
        return sheet

    def _sheet_file_path(self, sheet_info: SheetInfo) -> Path:
        if sheet_info.sheet_instance_path == "/":
            return self.root_schematic
        return (self.root_schematic.parent / sheet_info.sheet_file).resolve()

    def _is_passive(self, reference: str) -> bool:
        """Check if a component is passive based on reference prefix.

        Covers common passive component prefixes:
        R: Resistor, RN: Resistor network
        C: Capacitor, CP: Polarized capacitor
        L: Inductor, FB: Ferrite bead
        D: Diode, LED, TVS
        TP: Test point
        Y/X: Crystal / oscillator
        T: Transformer
        JP/J: Jumper
        """
        ref = str(reference or "").upper()
        # Multi-char prefixes first (to avoid RN matching R)
        if ref.startswith(("RN", "CP", "FB", "TP", "LED", "JP")):
            return True
        # Single-char prefixes
        if ref and ref[0] in ("R", "C", "L", "D", "Y", "X", "T", "J"):
            return True
        return False
