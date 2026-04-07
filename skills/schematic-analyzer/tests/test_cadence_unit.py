"""Unit tests for Cadence XML parser core functions.

Tests coordinate transforms, format detection, and symbol connection point
calculations without requiring large XML test data files.
"""

import sys
import tempfile
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tools.cadence.xml_parser import (
    _transform_pin_coords,
    _get_symbol_connection_point_static,
    is_cadence_xml,
)


class TestTransformPinCoords:
    """Test _transform_pin_coords — absolute pin coordinate calculation."""

    def test_no_transform(self):
        """Pin at part origin with no offset."""
        assert _transform_pin_coords(100, 200, 0, 0, 0, 0) == (100, 200)

    def test_offset_only(self):
        """Pin with hotpoint offset, no rotation/mirror."""
        assert _transform_pin_coords(100, 200, 10, 20, 0, 0) == (110, 220)

    def test_negative_offset(self):
        """Pin with negative offset."""
        assert _transform_pin_coords(100, 200, -5, -10, 0, 0) == (95, 190)


class TestGetSymbolConnectionPoint:
    """Test connection point calculation with rotation and mirror."""

    def test_no_rotation_no_mirror(self):
        """Default orientation — pin offset applied directly."""
        # pin_offset = (10, 0), loc = (100, 200)
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 0, 0)
        assert (x, y) == (110, 200)

    def test_rotation_90(self):
        """90° rotation: dx,dy = -dy,dx → (10,0) becomes (0,10)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 1, 0)
        assert (x, y) == (100, 210)

    def test_rotation_180(self):
        """180° rotation: dx,dy = -dx,-dy → (10,0) becomes (-10,0)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 2, 0)
        assert (x, y) == (90, 200)

    def test_rotation_270(self):
        """270° rotation: dx,dy = dy,-dx → (10,0) becomes (0,-10)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 3, 0)
        assert (x, y) == (100, 190)

    def test_mirror_no_rotation(self):
        """Mirror flips X: (10,0) becomes (-10,0)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 0, 1)
        assert (x, y) == (90, 200)

    def test_mirror_with_rotation_90(self):
        """Mirror + 90°: (10,0) → mirror → (-10,0) → rot90 → (0,-10)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 1, 1)
        assert (x, y) == (100, 190)

    def test_mirror_with_rotation_180(self):
        """Mirror + 180°: (10,0) → mirror → (-10,0) → rot180 → (10,0)."""
        x, y = _get_symbol_connection_point_static(100, 200, (10, 0), 2, 1)
        assert (x, y) == (110, 200)


class TestIsCadenceXml:
    """Test Cadence XML format detection."""

    def test_valid_cadence_xml(self):
        """File with dsn.xsd and <Design should be detected."""
        content = '<?xml version="1.0"?>\n<Design xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="dsn.xsd">'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            f.flush()
            assert is_cadence_xml(f.name) is True

    def test_non_cadence_xml(self):
        """Generic XML should not be detected."""
        content = '<?xml version="1.0"?>\n<root><item>test</item></root>'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            f.flush()
            assert is_cadence_xml(f.name) is False

    def test_nonexistent_file(self):
        """Missing file should return False."""
        assert is_cadence_xml("/nonexistent/file.xml") is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
