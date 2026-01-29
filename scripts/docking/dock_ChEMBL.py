# %%
import pandas as pd
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
RESULTS_DIR = PROJ_DIR / 'results'
DATASET_DIR = PROJ_DIR / "datasets"

sys.path.insert(0, str(SCRIPTS_DIR / "docking"))
from docking_fns import RunGNINA

sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from misc_functions import readConfigJSON

config_json = readConfigJSON(config_fpath=PROJ_DIR / 'config.json')
data_paths = config_json['data']
receptor_path = config_json["receptor"]

# train_df = pd.read_csv(
#     data_paths['it0_training_dock'],
#     index_col=False,
# )

hold_out_df = pd.read_csv(
#    data_paths['held_out_dock'],
    "/users/yhb18174/Prosperity_Partnership/docking/jaffer_kirsty_docking.csv",
    index_col=False,
)

hold_out_df['SMILES']

docking_dir = PROJ_DIR / "docking"
smi_ls = list(hold_out_df["SMILES"])
molid_ls = list(hold_out_df["ID"])

mp = RunGNINA(
    docking_dir=docking_dir,
    molid_ls=molid_ls,
    smi_ls=smi_ls,
    receptor_path=receptor_path,
    log_path="ChEMBL_docking"
)

sdfs = mp.processMols(use_multiprocessing=False)

# %%

job_ids = mp.submitMultipleJobs(run_hrs=2, run_mins=0)

# %%
ids, cnn_scores, aff_scores = mp.makeDockingCSV(save_data=True)

new_df = pd.DataFrame()
new_df["ID"] = molid_ls
new_df["SMILES"] = smi_ls
new_df["CNN_affinity"] = cnn_scores
new_df["Affinity(kcal/mol)"] = aff_scores

new_df.to_csv(
    PROJ_DIR / "docking" / "docking.csv",
    index="ID",
)

# %%
