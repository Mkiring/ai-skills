import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parents[4]
CLI_PATH = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts/schematic-cli.py"
DENALI_ROOT = REPO_ROOT / "data/denali/RK3576.kicad_sch"
E1005_ROOT = REPO_ROOT / "data/E1005_v01"
SCRIPTS_ROOT = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_help_only_exposes_light_refactor_command_surface() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "overview" in result.stdout
    assert "query" in result.stdout
    assert "cache" in result.stdout
    assert "analyze" not in result.stdout
    assert "trace" not in result.stdout
    assert "card" not in result.stdout
    assert "list" not in result.stdout
    assert "scope" not in result.stdout
    assert "interpret" not in result.stdout
    assert "export" not in result.stdout


def test_overview_renders_structural_sections_for_project() -> None:
    result = run_cli("overview", str(DENALI_ROOT))

    assert result.returncode == 0, result.stderr
    assert "Project Overview" in result.stdout
    assert "Page Navigation" in result.stdout
    assert "Core Component Candidates" in result.stdout
    assert "Root Schematic:" in result.stdout
    assert "Referenced Pages:" in result.stdout


def test_overview_json_output_contains_required_sections(tmp_path: Path) -> None:
    output_path = tmp_path / "overview.json"

    result = run_cli("overview", str(DENALI_ROOT), "--output", str(output_path))

    assert result.returncode == 0, result.stderr
    overview = load_json(output_path)
    assert overview["project_overview"]["root_schematic_filename"] == "RK3576.kicad_sch"
    assert overview["project_overview"]["referenced_page_count"] >= 1
    assert overview["page_navigation"]
    assert overview["core_component_candidates"]


def test_query_page_returns_structured_json_for_page_index() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--page", "1")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "page"
    assert payload["index"] == 1
    assert "components" in payload
    assert "nets" in payload
    assert all(component["mpn"] for component in payload["components"])


def test_query_component_returns_structured_json_for_reference() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "U36")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "component"
    assert payload["ref"] == "U36"
    assert payload["value"] == "FUSB302BMPX"
    assert payload["mpn"] == "FUSB302BMPX"
    assert "properties" in payload
    assert "nets" in payload
    assert "neighbors" in payload
    assert sorted(payload["neighbors"]) == ["shared_nets"]
    assert payload["neighbors"]["shared_nets"]

    for entry in payload["neighbors"]["shared_nets"]:
        assert sorted(entry) in (
            ["connected_refs", "cross_page", "fanout", "local_refs", "net"],
            ["connected_refs_sample", "cross_page", "fanout", "local_refs", "net", "truncated"],
        )
        assert "U36" not in entry.get("connected_refs", [])
        assert "U36" not in entry.get("connected_refs_sample", [])
        assert "U36" not in entry["local_refs"]
        assert isinstance(entry["cross_page"], bool)


def test_query_component_deduplicates_repeated_pin_net_pairs() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "U36")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    pairs = [(entry["name"], entry["pin"]) for entry in payload["nets"]]

    assert len(pairs) == len(set(pairs))


def test_query_component_default_omits_unconnected_nets_for_large_components() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "U23")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["query_type"] == "component"
    assert payload["ref"] == "U23"
    assert payload["nets"]
    assert all(not entry["name"].startswith("unconnected-") for entry in payload["nets"])


def test_query_component_full_restores_unconnected_nets() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "U23", "--full")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert any(entry["name"].startswith("unconnected-") for entry in payload["nets"])


def test_query_component_neighbors_are_sorted_by_fanout_then_name() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "U36")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    keys = [(entry["fanout"], entry["net"]) for entry in payload["neighbors"]["shared_nets"]]

    assert keys == sorted(keys)


def test_query_net_returns_structured_json_for_exact_name() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--net", "/FPCIO/FPC_I2C_SCL")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "net"
    assert payload["name"] == "/FPCIO/FPC_I2C_SCL"
    assert "pages" in payload
    assert "pins" in payload


def test_query_property_returns_structured_json_for_key() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--property", "MPN")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "property"
    assert payload["key"] == "MPN"
    assert payload["total"] >= payload["with_value"]
    assert payload["missing"] >= 0


def test_query_pattern_returns_structured_json_for_yaml() -> None:
    pattern_path = REPO_ROOT / ".claude/skills/schematic-analyzer/patterns/i2c.yaml"

    result = run_cli("query", str(DENALI_ROOT), "--pattern", str(pattern_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "pattern"
    assert payload["pattern_file"].endswith("i2c.yaml")
    assert "matches" in payload


def test_query_match_mode_finds_component_text_matches() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--component", "--match", "FUSB302")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "component"
    assert payload["search"] == "FUSB302"
    assert any(match["ref"] == "U36" for match in payload["matches"])


def test_query_match_mode_finds_net_text_matches() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--net", "--match", "I2C")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "net"
    assert payload["search"] == "I2C"
    assert payload["truncated"] is True
    assert payload["shown"] < payload["total"]
    assert any("I2C" in match["name"] for match in payload["matches"])


def test_query_match_mode_finds_regex_net_label_matches_in_e1005() -> None:
    result = run_cli("query", str(E1005_ROOT), "--net", "--match", "SDA|SCL|I2C", "--all")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "net"
    assert payload["search"] == "SDA|SCL|I2C"
    assert payload["total"] == 14
    assert payload["matches"] == [
        {
            "name": "MISC_I2C_SCL",
            "kind": "hierarchical_label",
            "occurrence_count": 4,
            "pages": ["Peripherals"],
            "mapped_to_net": False,
        },
        {
            "name": "MISC_I2C_SDA",
            "kind": "hierarchical_label",
            "occurrence_count": 4,
            "pages": ["Peripherals"],
            "mapped_to_net": False,
        },
        {
            "name": "BFG_I2C_SCL",
            "kind": "hierarchical_label",
            "occurrence_count": 1,
            "pages": ["Power"],
            "mapped_to_net": False,
        },
        {
            "name": "BFG_I2C_SDA",
            "kind": "hierarchical_label",
            "occurrence_count": 1,
            "pages": ["Power"],
            "mapped_to_net": False,
        },
        {
            "name": "TP_I2C_SCL",
            "kind": "hierarchical_label",
            "occurrence_count": 1,
            "pages": ["Epaper"],
            "mapped_to_net": False,
        },
        {
            "name": "TP_I2C_SDA",
            "kind": "hierarchical_label",
            "occurrence_count": 1,
            "pages": ["Epaper"],
            "mapped_to_net": False,
        },
        {
            "name": "BFG_I2C_SCL",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "BFG_I2C_SDA",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "MISC_I2C_SCL",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "MISC_I2C_SDA",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "TP_I2C_SCL",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "TP_I2C_SDA",
            "kind": "sheet_pin",
            "occurrence_count": 1,
            "pages": ["SCH_TOP"],
            "mapped_to_net": False,
        },
        {
            "name": "SCL",
            "kind": "local_label",
            "occurrence_count": 1,
            "pages": ["Epaper"],
            "mapped_to_net": False,
        },
        {
            "name": "SDA",
            "kind": "local_label",
            "occurrence_count": 1,
            "pages": ["Epaper"],
            "mapped_to_net": False,
        },
    ]
    assert all("naming_score" not in match for match in payload["matches"])
    assert all("pin_count" not in match for match in payload["matches"])


def test_query_match_mode_sorts_net_matches_by_naming_value_without_hiding_results() -> None:
    result = run_cli("query", str(E1005_ROOT), "--net", "--match", "SDA|SCL|I2C", "--all")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [match["name"] for match in payload["matches"][:6]] == [
        "MISC_I2C_SCL",
        "MISC_I2C_SDA",
        "BFG_I2C_SCL",
        "BFG_I2C_SDA",
        "TP_I2C_SCL",
        "TP_I2C_SDA",
    ]


def test_query_match_mode_respects_all_for_component_search() -> None:
    baseline = run_cli("query", str(DENALI_ROOT), "--component", "--match", "U")
    full = run_cli("query", str(DENALI_ROOT), "--component", "--match", "U", "--all")

    assert baseline.returncode == 0, baseline.stderr
    assert full.returncode == 0, full.stderr

    baseline_payload = json.loads(baseline.stdout)
    full_payload = json.loads(full.stdout)

    assert baseline_payload["query_type"] == "component"
    assert baseline_payload["truncated"] is True
    assert baseline_payload["shown"] < baseline_payload["total"]
    assert full_payload["truncated"] is False
    assert full_payload["shown"] == full_payload["total"]
    assert len(full_payload["matches"]) > len(baseline_payload["matches"])


def test_query_property_includes_truncation_metadata_and_all_override() -> None:
    baseline = run_cli("query", str(DENALI_ROOT), "--property", "Footprint")
    full = run_cli("query", str(DENALI_ROOT), "--property", "Footprint", "--all")

    assert baseline.returncode == 0, baseline.stderr
    assert full.returncode == 0, full.stderr

    baseline_payload = json.loads(baseline.stdout)
    full_payload = json.loads(full.stdout)

    assert baseline_payload["query_type"] == "property"
    assert "truncated" in baseline_payload
    assert "shown" in baseline_payload
    assert "total_values" in baseline_payload
    assert full_payload["truncated"] is False
    assert full_payload["shown"] == full_payload["total_values"]


def test_query_property_values_are_sorted_stably() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--property", "MPN", "--all")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    normalized = [
        (entry.get("mpn"), entry["refs"])
        for entry in payload["values"]
    ]
    assert normalized == sorted(normalized, key=lambda item: (item[0] is None, str(item[0])))


def test_query_net_match_results_are_sorted_by_naming_value() -> None:
    result = run_cli("query", str(DENALI_ROOT), "--net", "--match", "I2C", "--all")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["matches"][:5] == [
        {
            "name": "/SoM_MYC_LR3576/I2C3_M0.SCL",
            "kind": "net",
            "occurrence_count": 1,
            "pages": ["SoM_MYC_LR3576", "TYPEC#2", "TYPEC#3", "TYPEC#4"],
            "mapped_to_net": True,
            "pin_count": 5,
        },
        {
            "name": "/SoM_MYC_LR3576/I2C3_M0.SDA",
            "kind": "net",
            "occurrence_count": 1,
            "pages": ["SoM_MYC_LR3576", "TYPEC#2", "TYPEC#3", "TYPEC#4"],
            "mapped_to_net": True,
            "pin_count": 5,
        },
        {
            "name": "/SoM_MYC_LR3576/I2C4_SCL_M3",
            "kind": "net",
            "occurrence_count": 1,
            "pages": ["SoM_MYC_LR3576"],
            "mapped_to_net": True,
            "pin_count": 3,
        },
        {
            "name": "/SoM_MYC_LR3576/I2C4_SDA_M3",
            "kind": "net",
            "occurrence_count": 1,
            "pages": ["SoM_MYC_LR3576"],
            "mapped_to_net": True,
            "pin_count": 3,
        },
        {
            "name": "unconnected-(U23A-I2C8_SCL_M2-PadB71)",
            "kind": "net",
            "occurrence_count": 1,
            "pages": ["SoM_MYC_LR3576"],
            "mapped_to_net": True,
            "pin_count": 1,
        },
    ]


def test_cache_status_is_default_action() -> None:
    result = run_cli("cache", str(DENALI_ROOT))

    assert result.returncode == 0, result.stderr
    assert "Cache Status:" in result.stdout


def test_cache_clear_succeeds() -> None:
    result = run_cli("cache", str(DENALI_ROOT), "--clear")

    assert result.returncode == 0, result.stderr
    assert "Cache cleared" in result.stdout
