import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parents[4]
SCRIPTS_ROOT = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analyzer import SchematicAnalyzer
from tools.connectivity_builder import ConnectivityBuilder
from tools.project_indexer import ProjectIndexer
from tools.scope_resolver import ScopeResolver


CLI_PATH = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts/schematic-cli.py"
DENALI_DIR = REPO_ROOT / "data/denali"
DENALI_ROOT = DENALI_DIR / "RK3576.kicad_sch"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_scope_resolver_for_directory_returns_root_and_hierarchy_only() -> None:
    scope = ScopeResolver.resolve(DENALI_DIR)

    assert scope.root_schematic == DENALI_ROOT
    assert scope.project_name == "RK3576"
    assert scope.is_hierarchical is True
    assert all(path.parent == DENALI_DIR for path in scope.referenced_sheets)
    assert not hasattr(scope, "supplemental_sheets")


def test_project_index_keys_components_by_reference() -> None:
    index = ProjectIndexer().build(DENALI_ROOT)

    assert "U36" in index.components
    assert index.components["U36"].reference == "U36"
    assert index.components["U36"].sheet_path


def test_project_index_hierarchy_contains_root_and_child_pages() -> None:
    index = ProjectIndexer().build(DENALI_ROOT)

    assert index.hierarchy
    assert index.hierarchy[0].sheet_type == "root"
    assert any(sheet.sheet_type == "hierarchy" for sheet in index.hierarchy[1:])


def test_connectivity_builder_uses_net_name_as_root_identity() -> None:
    index = ProjectIndexer().build(DENALI_ROOT)
    graph = ConnectivityBuilder().build(index, DENALI_ROOT)

    assert graph.netlist_available is True
    assert "/FPCIO/FPC_I2C_SCL" in graph.all_nets
    assert graph.all_nets["/FPCIO/FPC_I2C_SCL"].net_name == "/FPCIO/FPC_I2C_SCL"
    assert graph.all_nets["/FPCIO/FPC_I2C_SCL"].net_type == "Unclassified"


def test_analyzer_overview_uses_hierarchy_scope_without_supplemental_pages() -> None:
    analyzer = SchematicAnalyzer(str(DENALI_ROOT))
    overview = analyzer.build_overview()

    page_files = {page["page_source_file"] for page in overview["page_navigation"]}
    assert "USB_MUX.kicad_sch" not in page_files


def test_overview_page_navigation_order_is_stable_and_root_first() -> None:
    analyzer = SchematicAnalyzer(str(DENALI_ROOT))

    first = analyzer.build_overview()
    second = analyzer.build_overview()

    assert first["page_navigation"] == second["page_navigation"]
    assert first["page_navigation"][0]["page_type"] == "root"
    indices = [page["page_index"] for page in first["page_navigation"]]
    assert indices == list(range(1, len(indices) + 1))


def test_overview_core_candidate_order_is_structurally_stable() -> None:
    analyzer = SchematicAnalyzer(str(DENALI_ROOT))

    overview = analyzer.build_overview()
    candidates = overview["core_component_candidates"]

    assert candidates
    assert [candidate["candidate_index"] for candidate in candidates] == list(range(1, len(candidates) + 1))
    ranked_keys = [
        (
            candidate["connected_net_count"],
            candidate["neighboring_symbol_count"],
            candidate["reference"],
        )
        for candidate in candidates
    ]
    assert ranked_keys == sorted(ranked_keys, reverse=True)


def test_overview_neighboring_symbol_count_matches_shared_net_peer_count() -> None:
    analyzer = SchematicAnalyzer(str(DENALI_ROOT))

    overview = analyzer.build_overview()
    phase_1 = analyzer._run_phase_1()
    component_nets = phase_1["component_nets"]

    for candidate in overview["core_component_candidates"][:5]:
        reference = candidate["reference"]
        pins = component_nets.get(reference, {}).get("pins", {})
        target_nets = {str(net_name) for net_name in pins.values() if str(net_name)}
        expected_neighbors = {
            other_ref
            for other_ref, entry in component_nets.items()
            if other_ref != reference
            and target_nets.intersection({str(net_name) for net_name in entry.get("pins", {}).values() if str(net_name)})
        }
        assert candidate["neighboring_symbol_count"] == len(expected_neighbors)


def test_removed_scope_command_is_not_available() -> None:
    result = run_cli("scope", str(DENALI_ROOT))

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
