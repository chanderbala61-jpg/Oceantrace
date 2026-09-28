"""
tests/test_sar_image_overlay.py
================================
Tests for build_sar_image_overlay():
  - WGS-84 bounds are correct and tightly aligned to the GeoTIFF
  - Returns a valid base64-encoded PNG data URI
  - CRS is EPSG:4326
  - Gracefully returns None for missing/invalid paths
  - Bounds are ordered [[south, west], [north, east]]
  - Overlay bounds contain the known spill centroid for scene 00955
"""

import os
import base64
import struct
import zlib
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENE_00955_PATH = os.path.join(REPO_ROOT, "extracted_dataset", "images", "00955.tif")
SCENE_00594_PATH = os.path.join(REPO_ROOT, "extracted_dataset", "images", "00594.tif")

# Known geographic facts about scene 00955 (Gulf of Mexico)
SCENE_00955_EXPECTED_BOUNDS = {
    "west": -94.93365127382808,
    "east": -94.7496763036404,
    "south": 26.05405691182433,
    "north": 26.238031882012006,
}
# Scene 00955 spill centroid (approximate) — must lie inside overlay bounds
SCENE_00955_CENTROID_LAT = 26.14   # approx centre
SCENE_00955_CENTROID_LON = -94.84  # approx centre


def _parse_png_ihdr(raw_bytes: bytes):
    """Returns (width, height) from PNG IHDR chunk."""
    assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG signature"
    # IHDR starts at byte 8: 4 length + 4 type + 13 data
    w = struct.unpack(">I", raw_bytes[16:20])[0]
    h = struct.unpack(">I", raw_bytes[20:24])[0]
    return w, h


@pytest.mark.skipif(
    not os.path.exists(SCENE_00955_PATH),
    reason="Scene 00955 GeoTIFF not present"
)
class TestSARImageOverlay00955:

    def _get_overlay(self):
        """Import build_sar_image_overlay from app module context."""
        import sys
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        # We test the pure function directly, bypassing Streamlit cache decorator
        import importlib
        import types
        # Monkeypatch st.cache_data to be a no-op for this test
        import streamlit as st
        original = st.cache_data
        st.cache_data = lambda **kw: (lambda f: f)
        try:
            import app as app_module
            importlib.reload(app_module)
            result = app_module.build_sar_image_overlay(SCENE_00955_PATH)
        finally:
            st.cache_data = original
        return result

    def test_returns_dict_not_none(self):
        """build_sar_image_overlay must return a dict for a valid path."""
        result = self._get_overlay()
        assert result is not None, "Expected dict, got None for valid GeoTIFF path"
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = self._get_overlay()
        assert result is not None
        assert "img_uri" in result
        assert "bounds" in result
        assert "crs" in result
        assert "image_path" in result

    def test_bounds_format(self):
        """bounds must be [[south, west], [north, east]] as Folium expects."""
        result = self._get_overlay()
        assert result is not None
        bounds = result["bounds"]
        assert len(bounds) == 2
        sw, ne = bounds
        assert len(sw) == 2 and len(ne) == 2
        south, west = sw
        north, east = ne
        assert south < north, "south must be less than north"
        assert west < east, "west must be less than east"

    def test_bounds_values_match_geotiff(self):
        """Bounds must closely match the known GeoTIFF extent."""
        result = self._get_overlay()
        assert result is not None
        south, west = result["bounds"][0]
        north, east = result["bounds"][1]
        assert abs(west - SCENE_00955_EXPECTED_BOUNDS["west"]) < 0.001
        assert abs(east - SCENE_00955_EXPECTED_BOUNDS["east"]) < 0.001
        assert abs(south - SCENE_00955_EXPECTED_BOUNDS["south"]) < 0.001
        assert abs(north - SCENE_00955_EXPECTED_BOUNDS["north"]) < 0.001

    def test_crs_is_wgs84(self):
        """CRS must be EPSG:4326 (WGS-84)."""
        result = self._get_overlay()
        assert result is not None
        assert "4326" in result["crs"], f"Expected EPSG:4326 CRS, got: {result['crs']}"

    def test_img_uri_is_base64_png(self):
        """img_uri must be a valid data:image/png;base64,... string."""
        result = self._get_overlay()
        assert result is not None
        uri = result["img_uri"]
        assert uri.startswith("data:image/png;base64,"), "URI must start with data:image/png;base64,"
        b64_part = uri[len("data:image/png;base64,"):]
        raw_bytes = base64.b64decode(b64_part)
        assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Decoded bytes must be a valid PNG"

    def test_png_dimensions_reasonable(self):
        """Decoded PNG must be at most 1024x1024 (downsampling applied)."""
        result = self._get_overlay()
        assert result is not None
        b64_part = result["img_uri"][len("data:image/png;base64,"):]
        raw_bytes = base64.b64decode(b64_part)
        w, h = _parse_png_ihdr(raw_bytes)
        assert w <= 1024 and h <= 1024, f"PNG is {w}x{h}, expected <=1024x1024"
        assert w > 0 and h > 0

    def test_spill_centroid_inside_bounds(self):
        """Known scene 00955 spill centroid must lie within the SAR overlay bounds."""
        result = self._get_overlay()
        assert result is not None
        south, west = result["bounds"][0]
        north, east = result["bounds"][1]
        assert south <= SCENE_00955_CENTROID_LAT <= north, (
            f"Centroid lat {SCENE_00955_CENTROID_LAT} not inside [{south}, {north}]"
        )
        assert west <= SCENE_00955_CENTROID_LON <= east, (
            f"Centroid lon {SCENE_00955_CENTROID_LON} not inside [{west}, {east}]"
        )

    def test_image_path_preserved(self):
        """image_path in result must match the input path."""
        result = self._get_overlay()
        assert result is not None
        assert result["image_path"] == SCENE_00955_PATH


class TestSARImageOverlayGracefulDegradation:

    def _call(self, path):
        import sys
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import streamlit as st
        original = st.cache_data
        st.cache_data = lambda **kw: (lambda f: f)
        try:
            import app as app_module
            import importlib
            importlib.reload(app_module)
            result = app_module.build_sar_image_overlay(path)
        finally:
            st.cache_data = original
        return result

    def test_none_path_returns_none(self):
        assert self._call(None) is None

    def test_empty_string_returns_none(self):
        assert self._call("") is None

    def test_nonexistent_path_returns_none(self):
        assert self._call("/nonexistent/path/to/scene.tif") is None
