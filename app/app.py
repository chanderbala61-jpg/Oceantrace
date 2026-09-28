"""
OceanTrace Streamlit Cloud Entrypoint Forwarder
===============================================
Ensures seamless deployment whether Streamlit Cloud is configured with
main file path 'app.py' or legacy path 'app/app.py'.
"""
import os
import sys
import runpy

APP_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

target_script = os.path.join(REPO_ROOT, "app.py")
runpy.run_path(target_script, run_name="__main__")
