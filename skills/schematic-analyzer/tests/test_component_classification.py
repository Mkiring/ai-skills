import sys
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parents[4]
SCRIPTS_ROOT = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from tools.core_ranker import rank_core_candidates
from tools.core_detector import (
    calculate_component_score,
    detect_core_components,
    get_main_controller,
)
from tools.role_classifier import classify_component


def test_core_detector_avoids_broad_prefix_false_positive_for_wireless_module() -> None:
    components = [
        {
            "reference": "U1",
            "value": "AP6256",
            "lib_id": "RF_Module:AP6256",
            "pins": [
                {"number": "1", "name": "SDIO_CLK"},
                {"number": "2", "name": "SDIO_CMD"},
                {"number": "3", "name": "WL_REG_ON"},
                {"number": "4", "name": "BT_UART_TXD"},
                {"number": "5", "name": "BT_UART_RXD"},
                {"number": "6", "name": "PCM_CLK"},
            ],
            "nets": [],
        },
        {
            "reference": "U2",
            "value": "XYZ1234",
            "lib_id": "MCU_Generic:QFN64_MCU",
            "pins": [
                {"number": "1", "name": "GPIO0"},
                {"number": "2", "name": "GPIO1"},
                {"number": "3", "name": "GPIO2"},
                {"number": "4", "name": "GPIO3"},
                {"number": "5", "name": "GPIO4"},
                {"number": "6", "name": "GPIO5"},
                {"number": "7", "name": "GPIO6"},
                {"number": "8", "name": "GPIO7"},
                {"number": "9", "name": "GPIO8"},
                {"number": "10", "name": "GPIO9"},
                {"number": "11", "name": "SWDIO"},
                {"number": "12", "name": "SWCLK"},
                {"number": "13", "name": "NRST"},
                {"number": "14", "name": "BOOT0"},
                {"number": "15", "name": "UART_TX"},
                {"number": "16", "name": "UART_RX"},
                {"number": "17", "name": "I2C_SCL"},
                {"number": "18", "name": "I2C_SDA"},
                {"number": "19", "name": "SPI_MOSI"},
                {"number": "20", "name": "SPI_MISO"},
                {"number": "21", "name": "SPI_SCK"},
                {"number": "22", "name": "USB_DP"},
                {"number": "23", "name": "USB_DM"},
                {"number": "24", "name": "ADC0"},
            ],
            "nets": [],
        },
    ]

    core_components = detect_core_components(components, top_n=2)
    assert all(component.reference != "U1" for component in core_components)
    assert get_main_controller(core_components).reference == "U2"


def test_role_classifier_uses_generic_library_and_pin_evidence_for_charger() -> None:
    component = {
        "reference": "U3",
        "value": "XYZ999",
        "lib_id": "Power_Management:Battery_Charger",
        "pins": [
            {"name": "VBUS"},
            {"name": "BAT"},
            {"name": "SYS"},
            {"name": "SW"},
            {"name": "PG"},
            {"name": "CE"},
            {"name": "SCL"},
            {"name": "SDA"},
        ],
        "nets": ["VBUS", "VBAT", "VSYS", "I2C_SCL", "I2C_SDA"],
    }

    role = classify_component(component)

    assert role.primary == "Power/Charger"
    assert role.confidence >= 0.8
    assert any("charger" in evidence.lower() for evidence in role.evidence)


def test_role_classifier_keeps_bus_membership_as_secondary_evidence() -> None:
    component = {
        "reference": "U9",
        "value": "UNKNOWN123",
        "lib_id": "Custom:QFN16",
        "pins": [
            {"name": "SCL"},
            {"name": "SDA"},
            {"name": "INT"},
            {"name": "ADDR"},
        ],
        "nets": ["I2C_SCL", "I2C_SDA", "SENSOR_INT"],
    }

    role = classify_component(component)

    assert role.primary == "Unknown"
    assert "I2C Participant" in role.secondary


def test_role_classifier_identifies_myc_lr3576_as_soc_module() -> None:
    component = {
        "reference": "U23",
        "value": "MYC-LR3576",
        "lib_id": "symbols:MYC_LR3576",
        "pins": [],
        "nets": [],
    }

    role = classify_component(component)

    assert role.primary == "Processor/SoC"


def test_core_score_ignores_role_and_signal_name_evidence() -> None:
    soc_like_component = {
        "reference": "U23",
        "value": "MYC-LR3576",
        "lib_id": "symbols:MYC_LR3576",
        "pins": [
            {"name": "DDR_DQ0"},
            {"name": "DDR_DQ1"},
            {"name": "I2C_SCL"},
            {"name": "I2C_SDA"},
        ],
        "nets": ["DDR0", "DDR1", "CTRL_SCL", "CTRL_SDA"],
    }
    generic_component = {
        "reference": "U99",
        "value": "CUSTOM_MODULE",
        "lib_id": "Custom:Module",
        "pins": [
            {"name": "PIN1"},
            {"name": "PIN2"},
            {"name": "PIN3"},
            {"name": "PIN4"},
        ],
        "nets": ["N1", "N2", "N3", "N4"],
    }

    assert calculate_component_score(soc_like_component) == calculate_component_score(generic_component)


def test_core_ranker_keeps_structural_identity_and_excludes_role_fields() -> None:
    components = [
        {
            "instance_id": "/SoM/U23",
            "sheet_instance_path": "/SoM",
            "reference": "U23",
            "value": "MYC-LR3576",
            "lib_id": "symbols:MYC_LR3576",
            "pins": [{"name": f"PIN{i}"} for i in range(1, 25)],
            "nets": [f"NET{i}" for i in range(1, 13)],
        },
        {
            "instance_id": "/USB/U5",
            "sheet_instance_path": "/USB",
            "reference": "U5",
            "value": "Bridge",
            "lib_id": "Custom:Bridge",
            "pins": [{"name": f"PIN{i}"} for i in range(1, 7)],
            "nets": [f"SIG{i}" for i in range(1, 3)],
        },
    ]

    ranked = rank_core_candidates(components, top_n=2)

    assert ranked[0].instance_id == "/SoM/U23"
    assert ranked[0].sheet_instance_path == "/SoM"
    assert ranked[0].reference == "U23"
    assert ranked[0].pin_count == 24
    assert ranked[0].net_count == 12
    assert not hasattr(ranked[0], "role")
    assert not hasattr(ranked[0], "evidence")
