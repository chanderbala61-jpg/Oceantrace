"""
Automated Test for OceanTrace Streamlit Demonstration Interface
===============================================================
Verifies that app.py initializes cleanly, runs without unhandled exceptions,
and renders core UI elements using Streamlit AppTest framework.
"""

import os
import unittest
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PY_PATH = os.path.join(REPO_ROOT, "app.py")


class TestStreamlitApp(unittest.TestCase):

    def test_app_loads_and_runs_without_exceptions(self):
        """Runs app.py headlessly and checks for exceptions."""
        at = AppTest.from_file(APP_PY_PATH, default_timeout=90)
        at.run()

        # Verify no unhandled exceptions thrown during render
        self.assertEqual(len(at.exception), 0, f"App threw exceptions: {at.exception}")

        # Verify sidebar loaded with radio navigation
        self.assertTrue(len(at.sidebar.radio) > 0, "Sidebar radio navigation missing")
        self.assertIn("1. Dashboard", at.sidebar.radio[0].options)

        # Verify main page content rendered
        self.assertTrue(len(at.markdown) > 0, "No markdown content rendered")


if __name__ == "__main__":
    unittest.main()
