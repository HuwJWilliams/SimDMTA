from workflow_class import SimDMTA
from pathlib import Path
import sys

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
DATASET_DIR = PROJ_DIR / 'datasets'
RESULTS_DIR =  PROJ_DIR / 'results'
SCRIPTS_DIR  = PROJ_DIR / 'scripts'

ORIGINAL_DIR = FILE_DIR.parents[2] / 'Recreating_DMTA'

sys.path.insert(0, str(SCRIPTS_DIR / 'models'))
from RF_class import RFModel

sys.path.insert(0, str(SCRIPTS_DIR / 'misc'))
from misc_functions import readConfigJSON

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

config_json = readConfigJSON(config_fpath=PROJ_DIR / 'config.json')
model_path = config_json['it0_model_dir']
data_paths = config_json['data']

# Pathing
full_data_fpath =  Path(data_paths["selection_pool_full"]).parent
full_data_fprefix = Path(data_paths["selection_pool_full"]).name

desc_fpath = Path(data_paths["selection_pool_desc"]).parent
desc_fprefix = Path(data_paths["selection_pool_desc"]).stem

docked_data_fpath = Path(data_paths["selection_pool_dock"]).parent 
docked_data_fprefix = Path(data_paths["selection_pool_dock"]).stem

init_model_dir = model_path

chosen_mol_file = RESULTS_DIR / run_name / 'chosen_mol.csv'

held_out_test_feats = Path(data_paths["held_out_desc"])
held_out_test_targs = Path(data_paths["held_out_dock"])
held_out_test_full = Path(data_paths["held_out_full"])

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
    id_prefix="CHEMBL",
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
