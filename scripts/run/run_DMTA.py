from workflow_class import SimDMTA
from pathlib import Path
import sys

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
DATASET_DIR = PROJ_DIR / 'datasets'
RESULTS_DIR =  PROJ_DIR / 'results'

ORIGINAL_DIR = FILE_DIR.parents[2] / 'Recreating_DMTA'

sys.path.insert(0, str(PROJ_DIR / 'scripts' / 'models'))

from RF_class import RFModel

# Iteration variables (adjusted within the .sh files)
start_iter = int(sys.argv[3])
total_iters = int(sys.argv[4])
n_cmpds = int(sys.argv[1])
selection_method = sys.argv[2]
if len(sys.argv) == 8 and sys.argv[7]: 
    run_name = f"{sys.argv[5]}_{n_cmpds}_{selection_method}_{sys.argv[7]}"
else:
    run_name = f"{sys.argv[5]}_{n_cmpds}_{selection_method}"
sel_size=float(sys.argv[6])


# Less Commonly Changed Variables
docking_column = "Affinity(kcal/mol)"
id_prefix = "PMG-"
max_confs = 100
log_level="DEBUG"

# Pathing
full_data_fpath =  DATASET_DIR / "test_dataset"
full_data_fprefix = "PMG_rdkit_*.csv.gz"

desc_fpath = DATASET_DIR / "test_dataset"
desc_fprefix = "PMG_rdkit_desc_*"

docked_data_fpath = DATASET_DIR / "test_dataset"
docked_data_fprefix = "PMG_docking_*"

init_model_dir = RESULTS_DIR / 'init_RF_model' / 'it0_test'

chosen_mol_file = RESULTS_DIR / run_name / 'chosen_mol.csv'

held_out_test_feats = DATASET_DIR / "test_dataset" / 'PMG_held_out_desc_trimmed.csv'
held_out_test_targs = DATASET_DIR / "test_dataset" / 'PMG_held_out_targ_trimmed.csv'
held_out_test_full = DATASET_DIR / "test_dataset" / 'PMG_rdkit_full.csv'

# Running the Workflow
run = SimDMTA(
    full_data_fpath=full_data_fpath,
    full_data_fprefix=full_data_fprefix,
    desc_data_fpath=desc_fpath,
    desc_data_fprefix=desc_fprefix,
    docked_data_fpath=docked_data_fpath,
    docked_data_fprefix=docked_data_fprefix,
    start_iter=start_iter,
    total_iters=total_iters,
    n_cmpds=n_cmpds,
    results_dir=RESULTS_DIR,
    init_model_dir=init_model_dir,
    chosen_mol_file=chosen_mol_file,
    selection_method=selection_method,
    run_name=run_name,
    docking_column=docking_column,
    max_confs=max_confs,
    sel_size=sel_size,
    log_level=log_level,
    docking_mins=5
)

run.runIterations(
    held_out_test_feats=held_out_test_feats, held_out_test_targs=held_out_test_targs, held_out_test_full=held_out_test_full
)
