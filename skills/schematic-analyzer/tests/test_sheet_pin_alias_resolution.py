from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = REPO_ROOT / ".claude/skills/schematic-analyzer/scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from tools.kicad.schematic_parser import SchematicParser


def test_find_connected_entities_includes_sheet_pin_alias_on_same_wire(tmp_path: Path) -> None:
    schematic = tmp_path / "parent.kicad_sch"
    schematic.write_text(
        """
(kicad_sch
  (version 20240101)
  (generator "test")
  (uuid "00000000-0000-0000-0000-000000000000")
  (sheet
    (at 30 20)
    (size 20 20)
    (property "Sheetname" "Child")
    (property "Sheetfile" "child.kicad_sch")
    (pin "EP_DCn" input
      (at 30 20 180)
    )
  )
  (label "GPIO16"
    (at 20 20 0)
  )
  (wire
    (pts
      (xy 10 20) (xy 30 20)
    )
  )
)
""".strip(),
        encoding="utf-8",
    )

    parser = SchematicParser(str(schematic), include_child_sheets=False)

    connected = parser.find_connected_entities({"GPIO16"})

    names = {(entry["kind"], entry["name"]) for entry in connected}
    assert ("local", "GPIO16") in names
    assert ("sheet_pin", "EP_DCn") in names
