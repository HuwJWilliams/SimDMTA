import pandas as pd
from glob import glob
import random
from pathlib import Path
import logging
import sys

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
# Misc
sys.path.insert(0, str(SCRIPTS_DIR / 'misc'))
from misc_functions import orderFiles


class MoleculeSelector:
    """
    Description
    -----------
    A class which holds all of the molecular selection methods:
    • Random selection of molecules
    • Highest/lowest value based on a provided column
    • Random choice from the highest/lowest value
    """

    def __init__(
        self,
        n_cmpds: int,
        preds_dir: str,
        chosen_mol_file: str,
        iteration: int,
        all_preds_prefix: str = "all_preds",
        logger=None,
        log_path=None,
        log_level: str="DEBUG"

    ):
        """ "
        Description
        -----------
        Initialising the class to globally hold the number of compounds, the prediction files and
        chosen_mol file. This will chose molecules not previously chosen by the algorithm, as
        indicated by compounds in the chosen_mol file.

        Parameters
        ----------
        n_cmpds (int)           Number of compounds to chose during the selection process
        preds_dir (str)         Directory containing all of the prediction files
        chosen_mol_file (str)   .csv file containing the IDs of previously chosen molecules
        iteration (int)         Iteration which the molecule selector is being called on
                                (used to update chosen_mol file)
        all_preds_prefix (str)  Prefix of the all_preds files which files have in common
                                (all_preds by default)

        Returns
        -------
        Class-wide/global instances of n_cmpds, preds_files, chosen_mol file, and iteration number

        """

        if logger:
            self.logger = logger
        else:
            # Setup a new logger
            logger_name = self.__class__.__name__
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.getLevelName(log_level.upper()))

            if not self.logger.hasHandlers():
                # Set default log file path
                log_path = Path(log_path) if log_path else Path(f"{logger_name}.log")

                file_handler = logging.FileHandler(log_path)
                file_handler.setLevel(logging.getLevelName(log_level.upper()))
                formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
                file_handler.setFormatter(formatter)

                self.logger.addHandler(file_handler)
                self.logger.propagate = False  # Don't pass logs to root

        self.logger.debug(f"Molecule selector logger initialised")

        self.n_cmpds = n_cmpds
        self.logger.debug(f"Files to select molecules from:\n{preds_dir}/{all_preds_prefix}*")
        

        print(f"Mol sel init 1: \n{preds_dir}/{all_preds_prefix}*")

        self.preds_files = orderFiles(glob(f"{preds_dir}/{all_preds_prefix}*"))

        print(f"Mol sel init 2: {self.preds_files}")

        self.chosen_mol_file = chosen_mol_file

        if chosen_mol_file:
            if not Path(self.chosen_mol_file).exists():
                self.chosen_mol = pd.DataFrame(columns=["Iteration"])
                self.chosen_mol.index.name = "ID"
                self.chosen_mol.to_csv(self.chosen_mol_file, index_label="ID")
            else:
                self.chosen_mol = pd.read_csv(chosen_mol_file, index_col="ID")

        self.it = iteration

    def random(self, n_mols:int=None):
        """
        Description
        -----------
        Function to select molecules at random across all of the prediction files

        Parameters
        ----------
        None

        Returns
        -------
        List of randomly chosen molecule IDs which are not in the chosen_mol file
        """

        self.logger.debug(f"Using random selection")

        if n_mols is None:
            n_mols=self.n_cmpds
        mols = []

        while len(mols) < n_mols:
            file = random.choice(self.preds_files)
            df = pd.read_csv(file, index_col="ID")
            id = random.choice(df.index.tolist())
            if id not in list(self.chosen_mol.index):
                mols.append(id)
            else:
                pass

        self.updateChosenMol(mols, iteration=self.it)

        return mols
    
    def best(self, column: str, ascending: bool, n_mols: int = None):
        """
        Select the best molecules globally across all prediction files based on the specified column.

        Parameters
        ----------
        column (str)        Column name to sort by
        ascending (bool)    If True, selects lowest values first (e.g. for uncertainty)
        n_mols (int)        Number of molecules to select (defaults to self.n_cmpds)

        Returns
        -------
        List of selected molecule IDs
        """
        self.logger.debug(f"Using best selection on column: {column}, ascending={ascending}")

        if n_mols is None:
            n_mols = self.n_cmpds

        # Read all data first, no head()
        df_list = [
            pd.read_csv(file, index_col="ID")
            for file in self.preds_files
        ]
        print(f"mol sel 1:\n{df_list}")

        full_df = pd.concat(df_list)
        print(f"mol sel 2:\n{full_df}")

        full_df.index = full_df.index.astype(str)
        full_df = full_df.sort_values(by=[column, full_df.index.name], ascending=[ascending, True])
        print(f"mol sel 3:\n{full_df}")


        self.logger.debug(f"Column '{column}': min={full_df[column].min()}, max={full_df[column].max()}, mean={full_df[column].mean():.4f}")

        # Filter out already selected
        mols = []
        for mol_id in full_df.index:
            if mol_id not in self.chosen_mol.index:
                mols.append(mol_id)
                self.logger.debug(f"Selected molecule: {mol_id}, {column} = {full_df.loc[mol_id, column]}")
            if len(mols) >= n_mols:
                break

        self.updateChosenMol(mols, iteration=self.it)
        return mols
        
    def randomInBest(self, column: str, ascending: bool, frac: float, n_mols: int=None):
        """
        Description
        -----------
        Function to choose molecules at random from the top % of molecules, e.g., randomly choosing moleucles within the top
        10 % of predicted potency.

        Parameters
        ----------
        column (str)        Name of column to sort molecules by
        ascending (bool)    Flag to swap how the molecules are sorted.
                            True = Lowest to highest (top to bottom)
                            False = Highest to lowest (top to bottom)
        frac (float)        Fraction of data to look over, (0.1 = 10 %, etc.)

        Returns
        -------
        List of molecules chosen at random within the top % of molecules. Choses molecules not already present in chosen_mol file
        """
        self.logger.debug(f"Using random in best selection")

        if n_mols is None:
            n_mols=self.n_cmpds

        mols = []
        total_mols = 0
        df_ls = [
            pd.read_csv(preds_file, index_col="ID").sort_values(
                by=column, ascending=ascending
            )
            for preds_file in self.preds_files
        ]

        for dfs in df_ls:
            total_mols += len(dfs)

        head_mols = int(total_mols * frac)

        top_df_ls = [
            df.sort_values(by=column, ascending=ascending).head(head_mols) for df in df_ls
        ]
        
        full_df = (
            pd.concat(top_df_ls)
            .sort_values(by=column, ascending=ascending)
            .head(total_mols)
        )

        while len(mols) < n_mols:
            id = random.choice(full_df.index.tolist())
            if id not in list(self.chosen_mol.index):
                mols.append(id)
            else:
                pass

        self.updateChosenMol(mols, iteration=self.it)

        return mols
    
    def hybrid(self, sel_method, frac):
        self.logger.debug(f"Using hybrid selection ({sel_method})")

        split_method = sel_method.split("_")
        sel_methods = split_method[:-1]
        sel_ratios = split_method[-1].split(":")
        total_mols = self.n_cmpds

        mols_selected = []

        if len(sel_methods) != len(sel_ratios):
            raise ValueError("Mismatch between number of selection methods and ratios")

        for method, ratio in zip(sel_methods, sel_ratios):
            ratio = float(ratio)
            if ratio < 1:
                n_mols = max(1, int(ratio * total_mols))
            else:
                n_mols = int((ratio / 10) * total_mols)

            if method == "mp":
                sel_idx = self.best(column="pred_Affinity(kcal/mol)", ascending=False, n_mols=n_mols)
            elif method == "mpo":
                sel_idx = self.best(column="MPO", ascending=True, n_mols=n_mols)
            elif method == "mu":
                sel_idx = self.best(column="Uncertainty", ascending=True, n_mols=n_mols)
            elif method == "rmp":
                sel_idx = self.randomInBest(column="pred_Affinity(kcal/mol)", ascending=False, frac=frac, n_mols=n_mols)
            elif method == "rmpo":
                sel_idx = self.randomInBest(column="MPO", ascending=True, frac=frac, n_mols=n_mols)
            elif method == "rmu":
                sel_idx = self.randomInBest(column="Uncertainty", ascending=True, frac=frac, n_mols=n_mols)
            elif method == "r":
                sel_idx = self.random(n_mols=n_mols)
            else:
                self.logger.warning("Unrecognised selection method...")
                sel_idx = []

            mols_selected.extend(sel_idx)

        self.updateChosenMol(mols_selected, iteration=self.it)
        return mols_selected
        

    def updateChosenMol(self, mol_ls: list, save: bool = True, iteration: int=None):
        """
        Description
        -----------
        Function to update and save the chosen_mol file with molecules chosen by the given selection method.
        Overwrites any previous entries from the current iteration.

        Parameters
        ----------
        mol_ls (list)       List of molecule IDs to enter into the chosen_mol file
        save (bool)         Flag to save the new chosen mol file

        Returns
        -------
        Newly updated chosen_mol file (if save=False)
        """
        self.logger.debug(f"Updating chosen mol file for iteration {iteration}")

        # Re-read in case file changed on disk
        if Path(self.chosen_mol_file).exists():
            self.chosen_mol = pd.read_csv(self.chosen_mol_file, index_col="ID")
        else:
            self.chosen_mol = pd.DataFrame(columns=["Iteration"])

        self.chosen_mol = self.chosen_mol[self.chosen_mol["Iteration"] != self.it]

        new_rows = pd.DataFrame({"Iteration": [iteration] * len(mol_ls)}, index=mol_ls)
        new_rows.index.name = "ID"

        self.chosen_mol = pd.concat([self.chosen_mol, new_rows])
        self.chosen_mol = self.chosen_mol.sort_values(by=["Iteration", "ID"])

        self.logger.debug(f"Selected {len(mol_ls)} molecules for iteration {iteration}")
        self.logger.debug(f"Total molecules after update: {self.chosen_mol.shape[0]}")

        if save:
            self.chosen_mol.to_csv(self.chosen_mol_file, index=True)
        else:
            return self.chosen_mol