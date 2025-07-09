# %%
import sys
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/models")
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/run")

from RF_class import PredictNewTestSet
import pandas as pd
from glob import glob
from pathlib import Path

path = "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/"
tft = path + 'PMG_held_out_desc_top.csv'
ttg = path + 'PMG_held_out_targ_top.csv'
tfl = path + 'PMG_rdkit_full.csv'

tg = path + 'PMG_held_out_targ_trimmed.csv'
ft = path + "PMG_held_out_desc_trimmed.csv"
fl = path +  "PMG_rdkit_full.csv"

# %%

# %%
# experiments = glob('/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/50_sel/202*')
# all_experiments = [str(Path(path).name) for path in experiments]

all_experiments = ["20250508_50_mu_mp_4:26"]
# PredictNewTestSet(
#     feats=ft,
#     targs=tg,
#     full_data=fl,
#     test_set_name = 'held_out',
#     experiment_ls=all_experiments,
#     results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/'
# )

# %%
PredictNewTestSet(
    feats=tft,
    targs=ttg,
    full_data=tfl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/50_sel/'
)


# %%
# experiments = glob('/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/10_sel/202*')
# all_experiments = [str(Path(path).name) for path in experiments]
# PredictNewTestSet(
#     feats=ft,
#     targs=tg,
#     full_data=fl,
#     test_set_name = 'held_out',
#     experiment_ls=all_experiments,
#     results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/10_sel/'
# )

# # %%
# PredictNewTestSet(
#     feats=tft,
#     targs=ttg,
#     full_data=tfl,
#     test_set_name = 'trimmed_held_out',
#     experiment_ls=all_experiments,
#     results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/10_sel/'
# )


# %%

import sys
from pathlib import Path
import pandas as pd

PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, PROJ_DIR / "scripts/run/")
from average_all import AverageAll

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import (
    get_sel_mols_between_iters,
    molid_to_smiles,
    molid_ls_to_smiles,
)

# avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/complete_archive/10_sel/')

# # # %%
# avg._average_experiment(exp_suffix="10_rmp", n_iters=150)
# avg._average_experiment(exp_suffix="10_rmu", n_iters=150)
# avg._average_experiment(exp_suffix="10_rmpo", n_iters=150)
# avg._average_experiment(exp_suffix="10_mp", n_iters=150)
# avg._average_experiment(exp_suffix="10_mpo", n_iters=150)
# avg._average_experiment(exp_suffix="10_r", n_iters=150)
# avg._average_experiment(exp_suffix="10_mu", n_iters=150)

# avg = AverageAll(
#     results_dir=str(PROJ_DIR) + "/results/rdkit_desc/complete_archive/50_sel/"
# )

# avg._average_experiment(exp_suffix="50_rmp", n_iters=30)
# avg._average_experiment(exp_suffix="50_rmu", n_iters=30)
# avg._average_experiment(exp_suffix="50_rmpo", n_iters=30)
# avg._average_experiment(exp_suffix="50_mp", n_iters=30)
# avg._average_experiment(exp_suffix="50_mpo", n_iters=30)
# avg._average_experiment(exp_suffix="50_r", n_iters=30)
# avg._average_experiment(exp_suffix="50_mu", n_iters=30)

# avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/complete_archive/mp_mu_hybrid/')

# avg._average_experiment(exp_suffix="50_mp_mu_2:8", n_iters=30)
# avg._average_experiment(exp_suffix="50_mp_mu_5:5", n_iters=30)
# avg._average_experiment(exp_suffix="50_mp_mu_8:2", n_iters=30)


# avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/complete_archive/rmp_rmu_hybrid/')

# avg._average_experiment(exp_suffix="50_rmp_rmu_2:8", n_iters=30)
# avg._average_experiment(exp_suffix="50_rmp_rmu_5:5", n_iters=30)
# avg._average_experiment(exp_suffix="50_rmp_rmu_8:2", n_iters=30)

# # # %%
# avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/complete_archive/diff_pool/')

# avg._average_experiment(exp_suffix='50_rmp_05', n_iters=30)
# avg._average_experiment(exp_suffix='50_rmp_025', n_iters=30)
# avg._average_experiment(exp_suffix='50_rmp_01', n_iters=30)
# avg._average_experiment(exp_suffix='50_rmp_005', n_iters=30)
# avg._average_experiment(exp_suffix='50_rmp_0025', n_iters=30)

# # # %%