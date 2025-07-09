import pandas as pd
import sys
from glob import glob

sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/docking/")
from docking_fns import Run_GNINA

df = pd.read_csv(
    "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_docking.csv",
    index_col=False,
)

docking_dir = "/users/yhb18174/Recreating_DMTA/docking/held_out_data/"
smi_ls = list(df["SMILES"])
molid_ls = list(df["ID"])

mp = Run_GNINA(
    docking_dir=docking_dir,
    molid_ls=molid_ls,
    smi_ls=smi_ls,
    receptor_path="/users/yhb18174/Recreating_DMTA/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt",
)

mp.ProcessMols(use_multiprocessing=True)

ids, cnn_scores, aff_scores = mp.SubmitJobs(run_hrs=0, run_mins=20)

# mol_dir_ls = [f'{docking_dir}{molid}' for molid in molid_ls]

ids, cnn_scores, aff_scores = mp.MakeCsv(save_data=True)

print(ids)
print(cnn_scores)
print(aff_scores)

new_df = pd.DataFrame()
new_df["ID"] = molid_ls
new_df["SMILES"] = smi_ls
new_df["CNN_affinity"] = cnn_scores
new_df["Affinity(kcal/mol)"] = aff_scores

new_df.to_csv(
    "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_docked.csv",
    index="ID",
)
