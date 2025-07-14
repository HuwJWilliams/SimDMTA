# %%
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'

sys.path.insert(0, str(SCRIPTS_DIR) / "models")
sys.path.insert(0, SCRIPTS_DIR / "run")

from RF_class import PredictNewTestSet

new_desc_fpath = ""
new_dock_fpath = ""
new_full_fpath = ""

# %%
all_experiments = []

# %%
PredictNewTestSet(
    feats=new_desc_fpath,
    targs=new_dock_fpath,
    full_data=new_full_fpath,
    test_set_name = 'new_predictions',
    experiment_ls=all_experiments,
    results_dir = str(PROJ_DIR / "results")
)