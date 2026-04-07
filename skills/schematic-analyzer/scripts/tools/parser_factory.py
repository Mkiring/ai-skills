"""Format detection and parser dispatch for multi-EDA support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def detect_format(path: str | Path) -> str:
    """Detect schematic format from file path.

    Returns:
        "kicad" for KiCad files (.kicad_sch, .kicad_pro)
        "cadence" for Cadence OrCAD XML exports
        Raises ValueError for unknown formats.
    """
    p = Path(path).resolve()

    if p.suffix == ".kicad_sch" or p.suffix == ".kicad_pro":
        return "kicad"

    if p.suffix.lower() == ".xml":
        from .cadence.xml_parser import is_cadence_xml
        if is_cadence_xml(p):
            return "cadence"

    # Directory: check contents
    if p.is_dir():
        if list(p.glob("*.kicad_sch")) or list(p.glob("*.kicad_pro")):
            return "kicad"
        # Check for Cadence XML files
        from .cadence.xml_parser import is_cadence_xml
        for xml_file in p.glob("*.xml"):
            if is_cadence_xml(xml_file):
                return "cadence"

    raise ValueError(f"Cannot detect schematic format for: {path}")


def get_schematic_parser(path: str | Path, **kwargs: Any):
    """Create the appropriate schematic parser for the given file.

    Returns a parser instance (CadenceXMLParser or KiCad SchematicParser)
    with a compatible interface.
    """
    fmt = detect_format(path)

    if fmt == "cadence":
        from .cadence.xml_parser import CadenceXMLParser
        return CadenceXMLParser(str(path), **kwargs)

    # Default: KiCad
    from .kicad.schematic_parser import SchematicParser
    return SchematicParser(str(path), **kwargs)
