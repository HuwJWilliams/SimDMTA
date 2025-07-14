import pandas as pd
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[2]
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

train_df = pd.read_csv(
    data_paths['it0_training_dock'],
    index_col=False,
)

hold_out_df = pd.read_csv(
    data_paths['held_out_dock'],
    index_col=False,
)

docking_dir = PROJ_DIR / "docking"
smi_ls = list(train_df["SMILES"])
molid_ls = list(train_df["ID"])

mp = RunGNINA(
    docking_dir=docking_dir,
    molid_ls=molid_ls,
    smi_ls=smi_ls,
    receptor_path=receptor_path,
)

mp.ProcessMols(use_multiprocessing=True)

ids, cnn_scores, aff_scores = mp.SubmitJobs(run_hrs=0, run_mins=20)

ids, cnn_scores, aff_scores = mp.MakeCsv(save_data=True)

new_df = pd.DataFrame()
new_df["ID"] = molid_ls
new_df["SMILES"] = smi_ls
new_df["CNN_affinity"] = cnn_scores
new_df["Affinity(kcal/mol)"] = aff_scores

new_df.to_csv(
    f"test_data_paths['it0_training_dock']",
    index="ID",
)
