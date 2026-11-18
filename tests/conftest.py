"""Shared pytest setup.

SO101_HOME must point at a temp dir before anything imports so101: so101.paths binds its paths at import time, and the
tests must never write to this repo's so101_config/. This repo's URDFs are symlinked into the temp home, so sim tests
use the real URDFs through so101's default paths, while the MuJoCo model is generated in the temp home.
"""

import os
import tempfile
from pathlib import Path

URDF_DIR = Path(__file__).resolve().parents[1] / "so101_config" / "urdf"

home = Path(tempfile.mkdtemp(prefix="so101-test-home-"))
if URDF_DIR.is_dir():
    (home / "urdf").symlink_to(URDF_DIR, target_is_directory=True)
os.environ["SO101_HOME"] = str(home)
