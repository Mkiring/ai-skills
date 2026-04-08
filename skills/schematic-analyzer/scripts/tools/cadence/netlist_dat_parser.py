r"""Cadence Allegro netlist (.dat) parser.

Parses pstxnet.dat and pstxprt.dat files exported from OrCAD/Allegro.
These files provide authoritative pin-net connectivity computed by the
OrCAD engine, which is more accurate than XML coordinate matching.

Key features:
- CDS_PINID extraction with escape sequence handling (\X\ -> X)
- Mux pin name resolution for SoC GPIO (| separated alternatives)
- Page information from pstxprt.dat for multi-page schematics
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def find_netlist_dir(root_path: Path) -> Optional[Path]:
    """Find the directory containing Allegro netlist files.

    Searches for common netlist directory names and pstxnet.dat file.

    Args:
        root_path: Root schematic directory or file to search from.

    Returns:
        Path to netlist directory if found, None otherwise.
    """
    # If root_path is a file, use its parent directory
    search_path = root_path.parent if root_path.is_file() else root_path

    # Also check for netlist_* pattern directories
    for child in search_path.iterdir():
        if child.is_dir() and child.name.startswith("netlist_"):
            dat_file = child / "pstxnet.dat"
            if dat_file.is_file():
                return child

    # Common directory names for Allegro netlists
    netlist_dir_names = [
        "netlist",
        "allegro",
        "pstxnet",
    ]

    for dir_name in netlist_dir_names:
        netlist_dir = search_path / dir_name
        if netlist_dir.is_dir():
            # Check for pstxnet.dat file
            dat_file = netlist_dir / "pstxnet.dat"
            if dat_file.is_file():
                return netlist_dir

    # Also check search_path directly
    dat_file = search_path / "pstxnet.dat"
    if dat_file.is_file():
        return search_path

    # Recursive search for pstxnet.dat (depth-limited)
    for dat_file in search_path.rglob("pstxnet.dat"):
        return dat_file.parent

    return None


def _select_mux_pin_name(pin_id: str, net_name: str) -> str:
    """Select the appropriate pin name from a mux table.

    SoC GPIO pins often have multiple functions separated by `|`.
    This function selects the most appropriate name based on the net name.

    Args:
        pin_id: Raw CDS_PINID value, possibly containing `|` separated mux options.
        net_name: The net name this pin is connected to.

    Returns:
        Selected pin name. If no match with net_name, returns the first
        non-empty option, or "GPIO" fallback.
    """
    if '|' not in pin_id:
        return pin_id

    # Split by | and clean up each option
    candidates = [p.strip() for p in pin_id.split('|')]

    # Filter out empty/dash placeholders
    valid_candidates = [c for c in candidates if c and c != '--']

    if not valid_candidates:
        return 'GPIO'

    # Try to match with net name first
    for candidate in valid_candidates:
        if candidate == net_name:
            return candidate

    # Fallback: return first valid option (often the GPIO name ends with _D)
    # Prefer GPIO over other functions as it's the safe default
    for candidate in valid_candidates:
        if candidate.endswith('_D'):
            return candidate

    return valid_candidates[0]


def _clean_pin_escape(pin_name: str) -> str:
    r"""Clean Cadence escape sequences from pin names.

    Cadence uses \X\ format to mark active-low signals (with overline).
    This function removes all backslash escapes.

    Args:
        pin_name: Raw pin name from CDS_PINID.

    Returns:
        Cleaned pin name with escape sequences removed.
    """
    return pin_name.replace('\\', '')


def parse_pstxnet(file_path: Path) -> dict[str, dict[str, str]]:
    r"""Parse Allegro pstxnet.dat file to extract pin-net mapping.

    The pstxnet.dat file contains authoritative connectivity data
    computed by the OrCAD engine. Each NODE_NAME entry maps a
    component reference and pin to a net name.

    File format:
        NET_NAME
        'UART6_RX_M0'
         '@...':C_SIGNAL='...';
        NODE_NAME	U7 1B5
         '@...': '\PINNAME':CDS_PINID='\PINID';

    Args:
        file_path: Path to pstxnet.dat file.

    Returns:
        Dictionary mapping {refdes: {pin_name: net_name}}.
        Pin names are extracted from CDS_PINID with proper escape
        sequence handling and mux resolution.
    """
    pin_net_map: dict[str, dict[str, str]] = {}

    if not file_path.is_file():
        return pin_net_map

    content = file_path.read_text(encoding='utf-8', errors='ignore')
    lines = content.split('\n')

    i = 0
    current_net = ""

    while i < len(lines):
        line = lines[i].strip()

        # Check for NET_NAME - the net name is on the next line in single quotes
        if line.startswith('NET_NAME'):
            # Net name is on the next line: 'netname'
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Extract net name from single quotes
                net_match = re.match(r"'([^']+)'", next_line)
                if net_match:
                    current_net = net_match.group(1)

        # Check for NODE_NAME - format: NODE_NAME refdes pin_number
        elif line.startswith('NODE_NAME'):
            # Parse: NODE_NAME U7 1B5
            node_match = re.match(r'NODE_NAME\s+(\S+)\s+(\S+)', line)
            if node_match:
                refdes = node_match.group(1)
                _pin_number = node_match.group(2)

                # Look for CDS_PINID in the next few lines
                # Format: '@...': 'pinname':CDS_PINID='pinid';
                # or: '@...':CDS_PINID='pinid';
                j = i + 1
                while j < len(lines) and j < i + 5:
                    check_line = lines[j]
                    pinid_match = re.search(r"CDS_PINID='([^']+)'", check_line)
                    if pinid_match:
                        raw_pin_id = pinid_match.group(1)

                        # Clean escape sequences (e.g., \FAULT1\ -> FAULT1)
                        pin_id = _clean_pin_escape(raw_pin_id)

                        # Resolve mux pin if needed (e.g., "FUNC1 | FUNC2 | GPIO")
                        if '|' in pin_id:
                            pin_id = _select_mux_pin_name(pin_id, current_net)

                        if refdes not in pin_net_map:
                            pin_net_map[refdes] = {}

                        pin_net_map[refdes][pin_id] = current_net
                        break
                    j += 1

        i += 1

    return pin_net_map


def parse_pstxprt(file_path: Path) -> dict[str, dict]:
    r"""Parse Allegro pstxprt.dat file to extract component page information.

    The pstxprt.dat file contains component-to-page mapping data.
    Each PART_NAME block includes the P_PATH which contains the page number.

    File format:
        PART_NAME
         C1 'CC_C0201_DISCRETE_100NF':;
        SECTION_NUMBER 1
         '@...': P_PATH='...\pageX_...'

    Args:
        file_path: Path to pstxprt.dat file.

    Returns:
        Dictionary mapping {refdes: {"page": page_name, "footprint": ..., ...}}.
    """
    page_info: dict[str, dict] = {}

    if not file_path.is_file():
        return page_info

    content = file_path.read_text(encoding='utf-8', errors='ignore')
    lines = content.split('\n')

    i = 0
    current_refdes = ""
    current_value = ""

    while i < len(lines):
        line = lines[i].strip()

        # Check for PART_NAME block start
        if line.startswith('PART_NAME'):
            # Next line has: REFDES 'VALUE':;
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                part_match = re.match(r"(\S+)\s+'([^']+)':", next_line)
                if part_match:
                    current_refdes = part_match.group(1)
                    current_value = part_match.group(2)

        # Look for P_PATH which contains page info
        if 'P_PATH=' in line:
            # Extract page number from P_PATH='...\pageX_...'
            page_match = re.search(r"page(\d+)", line)
            if page_match and current_refdes:
                page_num = page_match.group(1)
                page_info[current_refdes] = {
                    "primitive": current_value,
                    "page": f"page{page_num}",
                    "footprint": "",  # Not available in pstxprt.dat
                    "value": current_value,
                }
                # Reset after processing
                current_refdes = ""
                current_value = ""

        i += 1

    return page_info


def build_pin_net_map_from_dat(netlist_dir: Path) -> dict[str, dict[str, str]]:
    """Build pin-net map from Allegro netlist directory.

    Reads pstxnet.dat from the given directory and returns
    the pin-to-net connectivity mapping.

    Args:
        netlist_dir: Directory containing pstxnet.dat file.

    Returns:
        Dictionary mapping {refdes: {pin_name: net_name}}.
        Returns empty dict if file not found or parsing fails.
    """
    pstxnet_file = netlist_dir / "pstxnet.dat"

    if not pstxnet_file.is_file():
        return {}

    return parse_pstxnet(pstxnet_file)
