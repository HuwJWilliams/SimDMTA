# %%
import sys
from glob import glob
from pathlib import Path


PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, str(PROJ_DIR) + "/scripts/docking/")
from docking_fns import (
    WaitForDocking,
    Run_GNINA,
    GetUndocked,
    CleanFiles
)

# %%
CleanFiles()
# %%
