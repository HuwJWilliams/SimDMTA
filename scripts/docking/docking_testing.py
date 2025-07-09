# %%

import pandas as pd
from rdkit import Chem
from pathlib import Path
from multiprocessing import Pool
from glob import glob
import numpy as np
# from docking_fns import CalcMPO

# Import Openeye Modules
from openeye import oechem

# Muting GPU warning
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
from openeye import oequacpac, oeomega

PROJ_DIR = Path(__file__).parent.parent.parent

# %%
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
import shutil
PROJ_DIR = Path(__file__).parent.parent.parent


def CleanFiles(fpath: str, 
               fname: str = "PMG_docking_*.csv",
               docking_column: str = "Affinity(kcal/mol)",
               contaminants: list = ["PD"],
               replacement: str = "",
               index_col: str = 'ID',
               remove_dirs: bool = False,
               docking_dir: str = f"{str(PROJ_DIR)}/docking/PyMolGen/"):
    """
    Description
    -----------
    Function to remove any contaminants from files (e.g., 'PD', 'NaN', False, ...)
    
    Parameters
    ----------
    fpath (str)             Path to docking files
    fname (str)             Docking file name. Can be either generic (e.g., * or ?) or specific
    docking_column (str)    Column from which to remove contaminants
    contaminants (list)     Values to remove
    replacement (str)       Values to replace contaminants with
    index_col (str)         Name of index column
    remove_dirs (bool)      Whether to remove associated directories if contaminants are found
    docking_dir (str)       Path to docking directories

    Returns
    -------
    dict
        A dictionary where keys are filenames and values are lists of molecule IDs that contained 'PD'.
    """

    working_path = f"{fpath}/{fname}"
    pd_contaminated_molecules = {}

    # Find files matching pattern
    fpath_ls = glob(working_path) if "*" in fname or "?" in fname else [working_path]

    for path in fpath_ls:
        working_df = pd.read_csv(path, index_col=index_col)

        # Identify molecules where 'PD' is present
        pd_molecules = working_df[working_df.isin(["PD"]).any(axis=1)].index.tolist()
        pd_contaminated_molecules[Path(path).name] = pd_molecules

        # Replace contaminants
        working_df.replace({"PD": replacement}, inplace=True)

        # Save cleaned file
        working_df.to_csv(path, index_label=index_col)

        print(f"Processed file: {Path(path).name}")
        print(f" - Found and replaced {len(pd_molecules)} instances of 'PD'.\n")

    # Remove contaminated directories/files if required
    if remove_dirs:
        for molecule_id_list in pd_contaminated_molecules.values():
            for pd_mol in molecule_id_list:
                contaminated_tar_gz_file = Path(f"{docking_dir}/{pd_mol}.tar.gz")
                contaminated_dir = Path(f"{docking_dir}/{pd_mol}/")

                if contaminated_tar_gz_file.exists():
                    contaminated_tar_gz_file.unlink()
                    print(f"Removed file: {contaminated_tar_gz_file}")

                elif contaminated_dir.exists():
                    shutil.rmtree(contaminated_dir)
                    print(f"Removed directory: {contaminated_dir}")

    return pd_contaminated_molecules

# %%

CleanFiles(fpath=f"{PROJ_DIR}/datasets/PyMolGen/docking/")
# %%