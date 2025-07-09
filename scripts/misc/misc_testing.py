# %%
import sys

sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/misc/")
from misc_functions import (
    clean_chosen_mol, 
    remove_running_dirs,
)

# %%
# clean_chosen_mol(
#     experiment_ls=[
#         "20241002_10_mp",
#         "20241002_10_r",
#         "20241011_10_mpo",
#         "20241011_10_rmpo",
#         "20241015_10_rmp",
#         "20241002_10_mpo",
#         "20241002_10_rmp",
#         "20241011_10_r",
#         "20241015_10_mpo",
#         "20241015_10_rmpo",
#         "20241002_10_mu",
#         "20241002_10_rmpo",
#         "20241011_10_rmp",
#         "20241015_10_r",
#         "20241011_10_mp",
#         "20241011_10_mu",
#         "20241015_10_mp"
#     ],
#     experiment_dir="/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/",
#     n_mols=10,
# )
# # %%
# remove_running_dirs(
#     experiment_ls= [
#         "20241002_10_mp",
#         "20241002_10_r",
#         "20241011_10_mpo",
#         "20241011_10_rmpo",
#         "20241015_10_rmp",
#         "20241002_10_mpo",
#         "20241002_10_rmp",
#         "20241011_10_r",
#         "20241015_10_mpo",
#         "20241015_10_rmpo",
#         "20241002_10_mu",
#         "20241002_10_rmpo",
#         "20241011_10_rmp",
#         "20241015_10_r",
#         "20241011_10_mp",
#         "20241011_10_mu",
#         "20241015_10_mp"
#     ],
#     experiment_dir="/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/",

# )
# %%
CleanFiles(fname='PMG_docking_1_copy.csv')
# %%
