import json
import sys
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).absolute().parents[4]
SCRIPTS_ROOT = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analyzer import SchematicAnalyzer
from tools.core_detector import detect_core_components
from tools.core_ranker import rank_core_candidates
from tools.overlay_interpreter import OverlayInterpreter
from tools.pattern_loader import PatternLoader
from tools.role_classifier import classify_component


CLI_PATH = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts/schematic-cli.py"
DENALI_ROOT = REPO_ROOT / "data/denali/RK3576.kicad_sch"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_classify_component_uses_runtime_rule_overlay_for_mcp_mapping(tmp_path: Path) -> None:
    overlay = tmp_path / "custom_rules.yaml"
    overlay.write_text(
        "\n".join(
            [
                "mcp_category_map:",
                "  Custom Wireless Modules: Custom/WirelessModule",
            ]
        ),
        encoding="utf-8",
    )

    role = classify_component(
        {"reference": "U1", "value": "X1", "lib_id": "Custom:X1", "pins": [], "nets": []},
        mcp_info={"category": "Custom Wireless Modules", "subcategory": "Combo"},
        rule_paths=(str(overlay),),
    )

    assert role.primary == "Custom/WirelessModule"


def test_classify_component_uses_runtime_rule_overlay_for_mcp_subcategory_mapping(tmp_path: Path) -> None:
    overlay = tmp_path / "subcategory_rules.yaml"
    overlay.write_text(
        "\n".join(
            [
                "mcp_subcategory_map:",
                "  Custom Gauge: Custom/BatteryObserver",
            ]
        ),
        encoding="utf-8",
    )

    role = classify_component(
        {"reference": "U7", "value": "X7", "lib_id": "Custom:X7", "pins": [], "nets": []},
        mcp_info={"category": "Unmapped Devices", "subcategory": "Custom Gauge"},
        rule_paths=(str(overlay),),
    )

    assert role.primary == "Custom/BatteryObserver"


def test_classify_component_supports_runtime_required_signal_groups(tmp_path: Path) -> None:
    overlay = tmp_path / "video_bridge_rules.yaml"
    overlay.write_text(
        "\n".join(
            [
                "roles:",
                "  - role: Interface/Video-Bridge",
                "    min_score: 5.0",
                "    core_score: 0",
                "    required_groups:",
                "      - name: displayport_side",
                "        net_patterns:",
                "          - '(?i)DP_AUX'",
                "          - '(?i)DP_D[0-3]_[PN]'",
                "      - name: mipi_side",
                "        net_patterns:",
                "          - '(?i)MIPI_(?:CSI|DSI)'",
                "          - '(?i)CSI\\d*\\.'",
                "          - '(?i)DSI\\d*\\.'",
            ]
        ),
        encoding="utf-8",
    )

    role = classify_component(
        {
            "reference": "U5",
            "value": "CUSTOM_BRIDGE",
            "lib_id": "Custom:QFN64",
            "pins": [],
            "nets": [
                "/TYPEC/DP_AUX_P",
                "/TYPEC/DP_D0_P",
                "/SOC/MIPI_CSI0.D0_P",
                "/SOC/MIPI_CSI0.CK_P",
            ],
        },
        rule_paths=(str(overlay),),
    )

    assert role.primary == "Interface/Video-Bridge"
    assert role.source == "rule_engine"


def test_detect_core_components_uses_runtime_core_scoring_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "core_scoring.yaml"
    overlay.write_text(
        "\n".join(
            [
                "core_scoring:",
                "  minimum_score: 1",
                "  excluded_reference_families: []",
                "  signal_pin_patterns: []",
                "  pin_count_scores:",
                "    - minimum: 1",
                "      points: 7",
                "  net_count_scores:",
                "    - minimum: 1",
                "      points: 5",
                "  signal_pin_scores: []",
                "  transport_hint_bonus:",
                "    minimum_pin_count: 99",
                "    points: 0",
            ]
        ),
        encoding="utf-8",
    )

    core_components = detect_core_components(
        [
            {
                "reference": "X1",
                "value": "CustomCoreProbe",
                "lib_id": "Custom:Probe",
                "pins": [{"number": "1", "name": "FOO_SIG"}],
                "nets": ["NET1"],
            }
        ],
        rule_paths=(str(overlay),),
    )

    assert len(core_components) == 1
    assert core_components[0].reference == "X1"
    assert core_components[0].score >= 12


def test_core_ranker_ignores_runtime_core_scoring_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "core_scoring.yaml"
    overlay.write_text(
        "\n".join(
            [
                "core_scoring:",
                "  minimum_score: 1",
                "  excluded_reference_families: []",
                "  signal_pin_patterns: []",
                "  pin_count_scores:",
                "    - minimum: 1",
                "      points: 7",
                "  net_count_scores:",
                "    - minimum: 1",
                "      points: 5",
                "  signal_pin_scores: []",
                "  transport_hint_bonus:",
                "    minimum_pin_count: 99",
                "    points: 0",
            ]
        ),
        encoding="utf-8",
    )

    core_candidates = rank_core_candidates(
        [
            {
                "reference": "X1",
                "value": "CustomCoreProbe",
                "lib_id": "Custom:Probe",
                "pins": [{"number": "1", "name": "FOO_SIG"}],
                "nets": ["NET1"],
            }
        ],
        rule_paths=(str(overlay),),
    )

    assert len(core_candidates) == 1
    assert core_candidates[0].reference == "X1"


def test_analyzer_tracks_runtime_rule_overlay_sources(tmp_path: Path) -> None:
    overlay = tmp_path / "context_only.yaml"
    overlay.write_text("roles: []\n", encoding="utf-8")

    analyzer = SchematicAnalyzer(str(DENALI_ROOT), rule_paths=[str(overlay)])
    analysis = analyzer.analyze(force=True)

    assert analysis.meta["rule_paths"] == [str(overlay)]
    assert analysis.meta["context_signature"]


def test_analyzer_without_explicit_pattern_sources_does_not_load_builtin_patterns() -> None:
    analyzer = SchematicAnalyzer(str(DENALI_ROOT))
    analysis = analyzer.analyze(force=True)

    assert analysis.meta["pattern_sources"] == []
    assert analysis.meta["loaded_patterns"] == []
    assert analysis.meta["matched_patterns"] == []


def test_pattern_loader_without_explicit_sources_loads_nothing(tmp_path: Path) -> None:
    loader = PatternLoader(project_path=str(tmp_path))

    assert loader.load() == []


def test_cli_query_pattern_accepts_runtime_patterns_overlay(tmp_path: Path) -> None:
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()
    (pattern_dir / "custom_vbus.yaml").write_text(
        "\n".join(
            [
                "name: CustomVBUS",
                "category: runtime",
                'description: "Runtime-injected VBUS detector"',
                "",
                "signals:",
                "  - role: vbus",
                "    patterns:",
                '      - "(?i)^/?VBUS$"',
                '      - "(?i)^/.+VBUS"',
                "    required: true",
                "    group_key: false",
                "",
                "controller:",
                "  detect_by: none",
                "",
                "participants:",
                "  filter:",
                '    - reference_prefix: ["R", "C", "L", "D", "TP", "FB"]',
                "  exclude_controller: false",
                "",
                "extra_fields: {}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "query",
        str(DENALI_ROOT),
        "--pattern",
        str(pattern_dir / "custom_vbus.yaml"),
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["query_type"] == "pattern"
    assert payload["pattern_file"].endswith("custom_vbus.yaml")
    assert "matches" in payload


def test_query_pattern_uses_explicit_pattern_source_only(tmp_path: Path) -> None:
    explicit_pattern = tmp_path / "explicit_only.yaml"
    explicit_pattern.write_text(
        "\n".join(
            [
                "name: ExplicitOnly",
                "category: runtime",
                "signals:",
                "  - role: never_match",
                '    patterns: ["(?i)^THIS_WILL_NOT_MATCH$"]',
                "    required: true",
                "    group_key: false",
                "controller:",
                "  detect_by: none",
                "participants:",
                "  filter: []",
                "  exclude_controller: false",
                "extra_fields: {}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "query",
        str(DENALI_ROOT),
        "--pattern",
        str(explicit_pattern),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query_type"] == "pattern"
    assert payload["pattern_file"].endswith("explicit_only.yaml")
    assert payload["matches"] == []


def test_overlay_interpreter_returns_instance_scoped_roles_from_runtime_rules(tmp_path: Path) -> None:
    overlay = tmp_path / "roles.yaml"
    overlay.write_text(
        "\n".join(
            [
                "roles:",
                "  - role: Interface/Video-Bridge",
                "    min_score: 5.0",
                "    core_score: 0",
                "    required_groups:",
                "      - name: displayport_side",
                "        net_patterns:",
                "          - '(?i)DP_AUX'",
                "          - '(?i)DP_D[0-3]_[PN]'",
                "      - name: mipi_side",
                "        net_patterns:",
                "          - '(?i)MIPI_(?:CSI|DSI)'",
                "          - '(?i)CSI\\d*\\.'",
                "          - '(?i)DSI\\d*\\.'",
            ]
        ),
        encoding="utf-8",
    )

    interpreter = OverlayInterpreter(rule_paths=(str(overlay),))
    roles = interpreter.interpret_components(
        [
            {
                "instance_id": "/TYPEC/U5",
                "reference": "U5",
                "value": "CUSTOM_BRIDGE",
                "lib_id": "Custom:QFN64",
                "pins": [],
                "nets": [
                    "/TYPEC/DP_AUX_P",
                    "/TYPEC/DP_D0_P",
                    "/SOC/MIPI_CSI0.D0_P",
                    "/SOC/MIPI_CSI0.CK_P",
                ],
            }
        ]
    )

    assert set(roles) == {"/TYPEC/U5"}
    assert roles["/TYPEC/U5"].instance_id == "/TYPEC/U5"
    assert roles["/TYPEC/U5"].primary == "Interface/Video-Bridge"
    assert roles["/TYPEC/U5"].source == "rule_engine"


def test_overlay_interpreter_preserves_instance_identity_for_duplicate_refs() -> None:
    interpreter = OverlayInterpreter()

    roles = interpreter.interpret_components(
        [
            {
                "instance_id": "/Page1/U1",
                "reference": "U1",
                "value": "UNKNOWN_SENSOR",
                "lib_id": "Custom:QFN16",
                "pins": [{"name": "SCL"}, {"name": "SDA"}],
                "nets": ["I2C1_SCL", "I2C1_SDA"],
            },
            {
                "instance_id": "/Page2/U1",
                "reference": "U1",
                "value": "UNKNOWN_SENSOR",
                "lib_id": "Custom:QFN16",
                "pins": [{"name": "SCL"}, {"name": "SDA"}],
                "nets": ["I2C2_SCL", "I2C2_SDA"],
            },
        ]
    )

    assert set(roles) == {"/Page1/U1", "/Page2/U1"}
    assert roles["/Page1/U1"].reference == "U1"
    assert roles["/Page2/U1"].reference == "U1"
    assert roles["/Page1/U1"].instance_id != roles["/Page2/U1"].instance_id


def test_overlay_interpreter_returns_pattern_matches_with_instance_scoped_participants(tmp_path: Path) -> None:
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()
    (pattern_dir / "custom_i2c.yaml").write_text(
        "\n".join(
            [
                "name: CustomI2C",
                "category: bus",
                'description: "Runtime-injected I2C detector"',
                "",
                "signals:",
                "  - role: scl",
                '    patterns: ["(?i)^I2C1_SCL$"]',
                "    required: true",
                "    group_key: false",
                "  - role: sda",
                '    patterns: ["(?i)^I2C1_SDA$"]',
                "    required: true",
                "    group_key: false",
                "",
                "controller:",
                "  detect_by: core_component",
                "",
                "participants:",
                "  filter:",
                '    - reference_prefix: ["R", "C", "L", "D", "TP", "FB"]',
                "  exclude_controller: false",
                "",
                "extra_fields: {}",
            ]
        ),
        encoding="utf-8",
    )

    interpreter = OverlayInterpreter(project_path=str(tmp_path), pattern_sources=(str(pattern_dir),))
    pattern_matches = interpreter.interpret_subsystems(
        components=[
            {
                "instance_id": "/Core/U1",
                "reference": "U1",
                "value": "MCU",
                "lib_id": "Custom:MCU",
                "pins": [],
                "nets": ["I2C1_SCL", "I2C1_SDA"],
                "role": {"primary": "Processor/MCU"},
            },
            {
                "instance_id": "/Sensors/U7",
                "reference": "U7",
                "value": "SENSOR",
                "lib_id": "Custom:Sensor",
                "pins": [],
                "nets": ["I2C1_SCL", "I2C1_SDA"],
                "role": {"primary": "Sensor/IMU"},
            },
        ],
        nets=[
            {"name": "I2C1_SCL", "code": 1, "identity_space": "root"},
            {"name": "I2C1_SDA", "code": 2, "identity_space": "root"},
        ],
        core_ref="U1",
    )

    match = next(pattern_match for pattern_match in pattern_matches if pattern_match.pattern_name == "CustomI2C")
    assert match.pattern_name == "CustomI2C"
    assert match.signals["scl"].net_name == "I2C1_SCL"
    assert match.signals["scl"].identity_space == "root"
    assert match.controller["instance_id"] == "/Core/U1"
    assert {participant["instance_id"] for participant in match.participants} == {"/Core/U1", "/Sensors/U7"}


def test_pattern_loader_uses_only_runtime_sources_when_explicit_patterns_are_provided(tmp_path: Path) -> None:
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()
    (pattern_dir / "custom_only.yaml").write_text(
        "\n".join(
            [
                "name: CustomOnly",
                "category: runtime",
                "signals:",
                "  - role: clk",
                '    patterns: ["(?i)^CLK$"]',
                "    required: true",
                "    group_key: false",
                "controller:",
                "  detect_by: none",
                "participants:",
                "  filter: []",
                "  exclude_controller: false",
                "extra_fields: {}",
            ]
        ),
        encoding="utf-8",
    )

    loader = PatternLoader(project_path=str(tmp_path), extra_sources=(str(pattern_dir),))
    names = [pattern.name for pattern in loader.load()]

    assert names == ["CustomOnly"]


def test_removed_card_command_is_not_available() -> None:
    result = run_cli("card", str(REPO_ROOT / "data/denali"), "--component", "U17")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
