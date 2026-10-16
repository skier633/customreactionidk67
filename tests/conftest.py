# lets the tests import the project files (gestures.py, pen.py...) and gives them a clean folder
import os
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def project(tmp_path, monkeypatch):
    # a throwaway copy of the folders the app reads and writes, so tests never touch your real settings
    for folder in ["assets", "sounds"]:
        shutil.copytree(os.path.join(ROOT, folder), tmp_path / folder)
    shutil.copy(os.path.join(ROOT, "studio.html"), tmp_path / "studio.html")
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path
