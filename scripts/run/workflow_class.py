import pandas as pd
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
import re
from prettytable import PrettyTable
import logging
import numpy as np
from pathlib import Path
import sys
import time
import json

import warnings
warnings.filterwarnings("ignore")

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}

# Inserting paths to relevant functions
# Models
sys.path.insert(0, str(SCRIPTS_DIR / 'models'))
from RF_class import RFModel, predictNewTestSet

# Docking
sys.path.insert(0, str(SCRIPTS_DIR / 'docking'))
from docking_fns import RunGNINA, wait4Docking, getUndocked

# Dataset
sys.path.insert(0, str(SCRIPTS_DIR / 'dataset'))
from dataset_fns import DatasetAccessor

# Molecule Selection
sys.path.insert(0, str(SCRIPTS_DIR / 'mol_sel'))
from mol_sel_fns import MoleculeSelector

# Misc
sys.path.insert(0, str(SCRIPTS_DIR / 'misc'))
from misc_functions import molid2BatchNo, wait4Jobs, orderFiles, resolveFiles


class SimDMTA:
    def __init__(
        self,
        full_data_fpath: str,
        full_data_fprefix: str,
        desc_data_fpath: str,
        desc_data_fprefix: str,
        docked_data_fpath:str,
        docked_data_fprefix:str,
        start_iter: int,
        total_iters: int,
        n_cmpds: int,
        results_dir: str,
        init_model_dir: str,
        chosen_mol_file: str,
        selection_method: str,
        run_name: str,
        docking_column: str="Affinity(kcal/mol)",
        pred_column: str="pred_Affinity(kcal/mol)",
        max_confs: int = 100,
        id_prefix: str = "PMG-",
        n_cpus: int = 40,
        receptor_path: str = str(SCRIPTS_DIR / 'docking' / 'receptors' / '4bw1_5_conserved_HOH.pdbqt'),
        max_runtime: int = 60 * 60 * 168,
        max_it_runtime: int = 60 * 60 * 160,  # 168hours
        hyper_params: dict = {
            "rf__n_estimators": [400, 500],
            "rf__max_features": ["sqrt"],
            "rf__max_depth": [25, 50, 75, 100],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [2, 4, 8]
        },
        username: str = "",
        sel_size: int=0.1,
        
        log_level: str = "DEBUG",
        docking_mins: int=20,
        docking_hrs: int=0
    ):
        """
        Description
        -----------
        Class to run the full RecDMTA project

        Parameters
        ----------
        full_data_fpath : str
               Path to the full data from each dataset (crucially includes the PFI and oe_logp columns)
        full_data_fprefix (str)     Prefix for the full_data csv files e.g., PMG_rdkit
        desc_data_fpath (str)            Path to the descriptors set for the data
        desc_data_fprefix (str)          Prefix for the descriptor csv files e.g., PMG_rdkit_desc
        start_iter (int)            Iteration to start the runs on
        total_iters (int)           Total number of iterations to complete
        n_cmpds (int)               Number of compounds chosen to dock & add to the training set
        docked_fpath (str)           Path to the directory containing all GNINA outputs
        docked_data_dir (str)       Path to the true docking files for the selection pool
        results_dir (str)           Path to the drectory containing all results
        init_model_dir (str)        Path to the initial model directory containing the predictions and
                                    initially trained model
        chosen_mol_file (str)       Path to the chosen_mol file, if none present enter where you'd like
                                    the chosen mol file to be (end with the csv name too)
        selection_method (str)      Method used to select molecules for retraining:
                                            'r'     = random
                                            'mp'    = most potent
                                            'rmpo'  = random in most potent
                                            'mpo'   = lowest mpo
                                            'rmpo'  = random in lowest mpo
                                            'mu'    = most uncertain
        docked_data_fprefix  (str)  Docking score files with an '*' as a batch number replacement
                                    e.g., PMG_docking_*.csv
        run_name (str)              Name of the directory the run will be saved under
        docking_column  (str)       Column in the docking files which contain the docking infomation
        pred_column (str)           Column in the prediction files containing the ML prediction
        max_confs (int)             The maximum number of conformers (tautomers, enantiomers, etc.
                                    generated per molecule to input into the docking)
        id_prefix (str)             Prefix to molecule IDs
        n_cpus (int)                Number of CPUs to use during the job
        receptor_path (str)         Pathway to the receptor used for docking
        max_runtime (int)           Maximum time a job can be run
        max_it_runtime (int)        Maximim time an iteration can be run
        hyper_params (dict)         Hyperparameters to optimise the RF models on
        username (str)              Username on the OS to submit the job under
        sel_size (float)            Fraction of the selection pool to use in selection methods
        log_level (str)             Logging verbosity: DEBUG, INFO, WARNING, or ERROR

        Returns
        -------
        Initialised class
        """

        self.full_df = pd.DataFrame()

        # Making run directory
        self.results_dir = results_dir
        self.run_dir = results_dir / run_name

        # Creating link between it0 of this run and the initial model
        if not self.run_dir.exists():
            self.run_dir.mkdir()
            link_name = self.run_dir / "it0"
            link_name.symlink_to(Path(init_model_dir))

        parameter_dict = {
            "batch_size": n_cmpds,
            "selection_method": selection_method,
            "selection_pool_size": sel_size,
            "hyper_params": hyper_params,
            "docking_time": (docking_hrs * 60) + docking_mins
        }
        
        json_fpath = self.run_dir / "run_params.json"
    
        with open(json_fpath, "w") as f:
            json.dump(parameter_dict, f, indent=4)
        
        # Setting up logging
        log_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
        self.logger = logging.getLogger(f"SimDMTA_{run_name}")
        self.logger.setLevel(log_level)

        log_file = Path(results_dir) / run_name / "SimDMTA.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s | %(levelname)s | %(funcName)s | Line %(lineno)d | \n%(message)s\n'
                )
            )
        file_handler.setLevel(log_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)

        self.logger.propagate = False
        self.logger.info(f"+=========== SimDMTA ===========+")

        self.logger.info(f"Logging to: {log_file}")


        # Initialising path the full data files, obtaining the file list
        # and ordering them numerically
        self.full_fpath = Path(full_data_fpath)
        self.full_fprefix = full_data_fprefix
        self.full_flist = resolveFiles(str(self.full_fpath / self.full_fprefix), logger=self.logger)
        self.logger.debug(f"Complete molecule files: {str(self.full_fpath / (self.full_fprefix + '.csv'))}")

        self.desc_data_fpath = Path(desc_data_fpath)
        self.desc_data_fprefix = desc_data_fprefix
        self.desc_flist = resolveFiles(str(self.desc_data_fpath / (self.desc_data_fprefix + "*.csv*")), logger=self.logger)

        self.logger.debug(f"Complete descriptor files: {self.desc_data_fpath}/{self.desc_data_fprefix}.csv")

        self.docked_fpath = Path(docked_data_fpath)
        self.docked_data_fprefix = docked_data_fprefix
        self.docked_flist = resolveFiles(str(self.docked_fpath / (self.docked_data_fprefix + ".csv")), logger=self.logger)

        if len(self.docked_flist) == 1:
            self.logger.warning("Only one docking batch file found. Using default batch index: 0")
            self.docked_file_map = {1: self.docked_flist[0]}
            self.default_batch = 1
        else:
            self.docked_file_map = {
                int(re.findall(r"\d+", Path(f).stem.split("_")[-1])[0]): f
                for f in self.docked_flist
            }
            self.default_batch = None

        self.logger.debug(f"Docked files map: {self.docked_file_map}")

        # Setting class variables
        # Paths
        self.init_model_dir = init_model_dir
        self.chosen_mol_file = chosen_mol_file
        self.receptor_path = receptor_path
        self.prev_it_dir = init_model_dir   
        self.prev_pred_flist = resolveFiles(str(Path(init_model_dir) / "all_preds_*.csv.gz"), logger=self.logger)


        # Iteration settings
        self.start_iter = start_iter
        self.total_iters = total_iters
        self.n_cmpds = n_cmpds
        self.docking_column = docking_column
        self.n_cpus = n_cpus
        self.selection_method = selection_method
        self.id_prefix = id_prefix
        self.hyper_params = hyper_params
        self.max_confs = max_confs
        self.username = username
        self.sel_size = sel_size
        self.pred_column = pred_column
        self.docking_mins = docking_mins
        self.docking_hrs = docking_hrs
        

        # Timings
        self.time_ran = 0
        self.max_runtime = max_runtime
        self.max_it_runtime = max_it_runtime
        self.run_times = []
        self.avg_runtime = 0

    def selectCompounds(self, iteration: int, prev_it_dir: str):
        """
        Description
        -----------
        Function to select molecules from a the previous iterations directory and add them to the chosen_mol.csv
        file. Also creates a pd.DataFrame with the selected molecules, and the batch number they belong to.

        Parameters
        ----------
        iteration (int)         Iteration which you are on, this saves which iteration each molecule was selected
                                in the chosen_mol file
        prev_it_dir (str)       Pathway to the precious iteration directory to select the molecules from the
                                prediction files

        Returns
        -------
        A pandas.DataFrame object with the following structure:
                        ____ __________
                        | ID | batch_no |
                        |----|----------|
                        | x  |    1     |
                        |____|__________|
        """
        self.logger.info("Selecting compounds.")
        self.logger.debug(f"Current iteration: {iteration}")
        self.logger.debug(f"Previous iteration dir: {prev_it_dir}")

        # Initialising molecule selection class
        self.sel = MoleculeSelector(
            n_cmpds=self.n_cmpds,
            preds_dir=prev_it_dir,
            chosen_mol_file=self.chosen_mol_file,
            iteration=iteration,
            logger=self.logger
        )
        
        self.logger.debug(f"Selecting {self.n_cmpds} molecules")

        # Selecting molecules
        if self.selection_method == "r":
            sel_idx = self.sel.random()

        elif self.selection_method == "mp":
            sel_idx = self.sel.best(column=self.pred_column, ascending=False)

        elif self.selection_method == "rmp":
            sel_idx = self.sel.randomInBest(
                column=self.pred_column, ascending=False, frac=self.sel_size
            )

        elif self.selection_method == "mu":
            sel_idx = self.sel.best(column="Uncertainty", ascending=False)

        elif self.selection_method == "mpo":
            sel_idx = self.sel.best(column="MPO", ascending=True)

        elif self.selection_method == "rmpo":
            sel_idx = self.sel.randomInBest(column="MPO", ascending=True, frac=self.sel_size)

        elif self.selection_method == "rmu":
            sel_idx = self.sel.randomInBest(column="Uncertainty", ascending=False, frac=self.sel_size)

        # For testing purposes
        elif self.selection_method == "test":
            sel_idx = ["PMG-31895", "PMG-27063"]
        
        # Hybrid methods
        elif ":" in self.selection_method:
            sel_idx = self.sel.hybrid(sel_method=self.selection_method, frac=self.sel_size)

        self.logger.debug(f"Molecules selected: {sel_idx}")

        print(f"Selection index:\n{sel_idx}")

        # Making a table for the selected mols
        self.df_select = pd.DataFrame(data=[], columns=[], index=sel_idx)
        print(f"1:\n{self.df_select}")

        self.df_select.index.rename("ID", inplace=True)
        print(f"2:\n{self.df_select}")

        self.df_select["batch_no"] = [
            molid2BatchNo(
                molid,
                self.id_prefix,
                str(Path(prev_it_dir) / "all_preds_*.csv.gz"),
                file_ls=self.prev_pred_flist,
                logger=self.logger
            ) if self.default_batch is None else self.default_batch
            for molid in self.df_select.index
        ]
        print(f"3:\n{self.df_select}")


        table = PrettyTable()
        table.field_names = ["ID"] + list(self.df_select.columns)
        for row in self.df_select.itertuples(index=True):
            table.add_row(row)
        

        self.logger.info("Molecules selected.\n" + str(table))

        print(f"4:\n{self.df_select}")
        return self.df_select

    def _submitJobsWrapper(self, args):
        """
        Wrapper for multiprocessing of job submission
        """

        batch_no, idxs_in_batch = args

        docking_score_batch_file = self.docked_file_map[batch_no]

        da = DatasetAccessor(
            original_path=docking_score_batch_file,
            wait_time=30,
            logger=self.logger

        )

        # Obtain exclusive access to the docking file
        docking_file = da.getExclusiveAccess()
        if docking_file is None:
            self.logger.warning(f"Failed to access file:\n{docking_score_batch_file}")
            self.logger.warning(f"Redocking of IDs:\n{idxs_in_batch} required")

        try:
            dock_df = pd.read_csv(docking_file, index_col=0)
        except UnicodeDecodeError:
            dock_df = pd.read_csv(docking_file, index_col=0, compression="gzip")

        # Isolating the molecule ids which have not already been docked or in the process 
        # of being docked
        for_docking = getUndocked(
            dock_df=dock_df,
            idxs_in_batch=idxs_in_batch,
            scores_col=self.docking_column,
            logger=self.logger
        )

        if "SMILES" not in for_docking.columns and "SMILES" in dock_df.columns:
            for_docking["SMILES"] = dock_df.loc[for_docking.index, "SMILES"]

        if for_docking.empty:
            self.logger.debug(f"No molecules to dock in batch {batch_no}...")
            da.releaseFile()
            return None, None, docking_score_batch_file, [], idxs_in_batch

        # Change docking value fr each molecule being docked as 'PD' (pending)
        da.editDF(
            column_to_edit=self.docking_column,
            idxs_to_edit=for_docking.index,
            vals_to_enter=["PD" for idx in for_docking.index],
        )

        # Releases exclusive access on file so parallel runs can access it
        da.releaseFile()

        self.logger.info(
            "** Docking compounds: " + ", ".join(for_docking.index.tolist()))

        molid_ls = []
        smi_ls = []

        for molid, smi in for_docking["SMILES"].items():
            molid_ls.append(molid)
            smi_ls.append(smi)

        # Initialising the docker
        docker = RunGNINA(
            docking_dir=self.docked_fpath,
            molid_ls=molid_ls,
            smi_ls=smi_ls,
            receptor_path=self.receptor_path,
            max_confs=self.max_confs,
            docking_general_file=str(self.docked_fpath / (self.docked_data_fprefix + ".csv")),
            logger=self.logger
        )

        # Creating sdfs with numerous conformers and adjusting for pH 7.4
        docker.processMols(use_multiprocessing=True)

        # Docking the molecules and saving scores in for_docking
        job_ids = docker.submitMultipleJobs(run_hrs=self.docking_hrs, run_mins=self.docking_mins, use_multiprocessing=True)

        if not job_ids:
            self.logger.warning(f"No docking jobs were submitted for: {for_docking.index.tolist()}")
            da = DatasetAccessor(
                original_path=docking_score_batch_file,
                wait_time=30,
                logger=self.logger
            )
            da.editDF(
                column_to_edit=self.docking_column,
                idxs_to_edit=for_docking.index,
                vals_to_enter=[""] * len(for_docking.index),
            )

        return docker, job_ids, docking_score_batch_file, molid_ls, idxs_in_batch

    def _dockingScoreRetrieval(
        self,
        dock_scores_ls: list,
        docking_batch_file: str,
        mols_to_edit_ls: list,
        idxs_in_batch: list,
    ):
        """
        Wrapper for multiprocessing the retrieval of docking scores
        """

        da = DatasetAccessor(
            original_path=docking_batch_file,
            wait_time=30,
            logger=self.logger
        )

        if mols_to_edit_ls:
            da.getExclusiveAccess()

            da.editDF(
                column_to_edit=self.docking_column,
                idxs_to_edit=mols_to_edit_ls,
                vals_to_enter=dock_scores_ls,
            )

            da.releaseFile()

            wait4Docking(
                docking_batch_file,
                idxs_in_batch=idxs_in_batch,
                scores_col=self.docking_column,
                check_interval=60,
                logger=self.logger
            )

        file_accessed = False
        p = Path(docking_batch_file)
        while not file_accessed:
            try:
                if p.exists() and p.stat().st_size > 0:
                    try:
                        batch_dock_df = pd.read_csv(str(p), index_col=0)
                    except UnicodeDecodeError as e:
                        batch_dock_df = pd.read_csv(str(p), index_col=0)

                    file_accessed = True
                else:
                    self.logger.warning(f"File {p} exists but is empty. Waiting...")
                    time.sleep(30)
            except FileNotFoundError:
                self.logger.warning(f"File {p} not found. Waiting for it to become accessible...")
                time.sleep(30)
            except pd.errors.EmptyDataError:
                self.logger.warning(f"File {p} is empty. Waiting for data to be written...")
                time.sleep(30)

        batch_dock_df = batch_dock_df.loc[idxs_in_batch]
        return batch_dock_df

    def runDocking(
        self,
    ):
        """
        Description
        -----------
        Function to run the docking portion of the workflow

        Parameters
        ---------
        None

        Returns
        -------
        pd.DataFrame object containing the docking information for the ids selected
        """
        
        # Arguments for _submit_jobs_wrapper
        sjw_args = [
            (batch_no, idxs_in_batch)
            for batch_no, idxs_in_batch in (
                self.df_select.reset_index()
                .groupby("batch_no")["ID"]
                .apply(list)
                .items()
            )
        ]

        # Getting all job ids
        all_job_id_ls = []
        initialised_dockers = []
        all_docking_score_batch_files = []
        all_molid_ls = []
        all_idxs_in_batch = []
        all_dock_scores_ls = []

        self.fin_dock_df = pd.DataFrame()

        for args in sjw_args:
            print(args)
            docker, job_ids, ds_batch_file, mols_to_edit_ls, idx_ls = (
                self._submitJobsWrapper(args)
            )

            if docker is not None:
                initialised_dockers.append(docker)
                all_job_id_ls.extend(job_ids)
                all_docking_score_batch_files.append(ds_batch_file)
                all_molid_ls.append(mols_to_edit_ls)
                all_idxs_in_batch.append(idx_ls)

            else:
                try:
                    docked_df = pd.read_csv(ds_batch_file, index_col="ID")
                except UnicodeDecodeError as e:
                    docked_df = pd.read_csv(ds_batch_file, index_col="ID")

                all_dock_scores_ls.append(docked_df[self.docking_column].loc[idx_ls])
                all_idxs_in_batch.append(idx_ls)
                all_molid_ls.append(mols_to_edit_ls)
                all_docking_score_batch_files.append(ds_batch_file)

        if all_job_id_ls:
            wait4Jobs(all_job_id_ls, logger=self.logger)

        for docker in initialised_dockers:
            molids, top_cnn_scores, top_aff_scores = docker.makeDockingCSV()
            all_dock_scores_ls.append(top_aff_scores)
            docker.compressDockingFiles()

        dsr_args = [
            (docking_scores_ls, docking_score_batch_file, molids, idxs_in_batch)
            for docking_scores_ls, docking_score_batch_file, molids, idxs_in_batch in zip(
                all_dock_scores_ls,
                all_docking_score_batch_files,
                all_molid_ls,
                all_idxs_in_batch,
            )
        ]

        # Retrieve docking scores
        with Pool() as pool:
            results = pool.starmap(self._dockingScoreRetrieval, dsr_args)

        self.fin_dock_df = pd.concat(results, axis=0)

        missing_ids = set(self.df_select.index) - set(self.fin_dock_df.index)
        if missing_ids:
            self.logger.warning(f"Missing IDs found:\n{missing_ids}")
            for mid in missing_ids:
                try:
                    batch_no = molid2BatchNo(
                        molid=mid,
                        prefix=self.id_prefix,
                        dataset_file=str(self.docked_fpath / (self.docked_data_fprefix + "*.csv")),
                        file_ls=self.docked_flist
                    ) if self.default_batch is None else self.default_batch               
                    dock_batch_csv = self.docked_file_map[batch_no]
                    dock_batch_df = pd.read_csv(dock_batch_csv, index_col="ID", dtype=str)

                    if mid in dock_batch_df.index:
                        row = dock_batch_df.loc[[mid]]
                        self.fin_dock_df = pd.concat([self.fin_dock_df, row], axis=0)
                        self.logger.warning(f"Updated missing ID: {mid}")
                    else:
                        self.logger.warning(f"ID {mid} not found in docking CSV: {dock_batch_csv}")

                except Exception as e:
                    self.logger.warning(f"Error updating missing ID {mid}:\n{e}")

        self.logger.info("Docking complete.")

        return self.fin_dock_df.loc[self.df_select.index]

    def updateTrainingSet(self):
        """
        Description
        -----------
        Function to obtain the previous training data and ammend the new, chosen molecules to it

        Parameters
        ----------
        None

        Returns
        -------
        x2 pd.DataFrame objects:      1- The dataframe of the updated target values
                                      2- The dataframe of the updated feature set

        """
        self.logger.info("Updating training set.")

        # Obtaining the training data from the previous iteration
        training_dir = Path(self.prev_it_dir)/ "training_data"

        prev_feats = pd.read_csv(
            training_dir / "training_features.csv.gz",
            index_col="ID",
        )
        prev_targs = pd.read_csv(
            training_dir / "training_targets.csv.gz", index_col="ID"
        )

        # Saving the columns which the previous iteration was trained on
        # (used to make sure the models are trained on the same set of features)
        self.prev_feat_cols = prev_feats.columns.tolist()

        # Dropping any molecules which failed to dock
        self.fin_dock_df = self.fin_dock_df.dropna(subset=[self.docking_column])

        # Getting the molecule IDs and their batch number to extract data from
        ids = self.fin_dock_df.index
        batch_nos = [
            molid2BatchNo(
                molid,
                self.id_prefix,
                str(self.desc_data_fpath / (self.desc_data_fprefix + "*.csv*")),
                file_ls=self.desc_flist,
            ) if self.default_batch is None else self.default_batch
            for molid in ids
        ]

        batch_to_ids = defaultdict(list)

        for id, batch_no in zip(ids, batch_nos):
            batch_to_ids[batch_no].append(id)

        added_desc = pd.DataFrame(columns=prev_feats.columns)

        # Creating a new df with all of the new data needed
        for batch in batch_to_ids:
            ids_in_batch = batch_to_ids[batch]
            desc_csv = str(self.desc_data_fpath / (self.desc_data_fprefix + '.csv'))
            desc_csv = desc_csv.replace("*", str(batch))
            desc_df = pd.read_csv(
                desc_csv, index_col="ID", usecols=prev_feats.reset_index().columns
            )
            desc_df = desc_df.loc[ids_in_batch]
            added_desc = pd.concat([added_desc, desc_df], axis=0)

        # Adding new rows onto the previous training data sets
        self.updated_feats = pd.concat([prev_feats, added_desc], axis=0)
        self.updated_targs = pd.concat(
            [prev_targs, self.fin_dock_df[[self.docking_column]]], axis=0
        )
        
        self.logger.info("Training set updated.")

        return self.updated_targs, self.updated_feats

    def _predict4Files(self, args: list):
        """
        Description
        -----------
        Function to wrap the predictive model in to allow for multiprocessing

        Parameters
        ----------
        args (list)     List containing the following parameters:
                        index        (number to give the suffix label the preds file)
                        desc_file    (descriptor file, given as a pathway)
                        full_file    (full file, given as a pathway)
                        model        (loaded prediction model class)

        Returns
        -------
        None
        """
        index, desc_file, full_file, model = args
        self.logger.debug(f"Predicting on descriptor file: {desc_file} and full file: {full_file}")
        
        try:
            feats = pd.read_csv(desc_file, index_col="ID")
            self.logger.debug(f"Features shape: {feats.shape}, columns: {list(feats.columns)}")
            
            model.predict(
                feats=feats[self.prev_feat_cols],
                save_preds=True,
                preds_save_path=self.it_dir,
                preds_filename=f"all_preds_{index+1}",
                final_rf=self.it_dir / "final_model.pkl",
                full_data_fpath=full_file,
            )
            self.logger.info(f"Prediction for {desc_file} completed successfully.")

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            self.logger.error(f"Error during prediction for {desc_file}:\n{tb_str}")

    def retrainAndPredict(self, feats: pd.DataFrame, targs: pd.DataFrame):
        """
        Description
        -----------
        Function to firstly retrain a new model using the updated data from UpdateTrainingSet() function
        Then uses the trained to predict docking scores

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.logger.info("Retraining model and making new predictions.")

        # Initialising the RF class
        model = RFModel(docking_column=self.docking_column, logger=self.logger)

        # Training on updated features
        rf = model.trainRegressor(
            search_type="grid",
            hyper_params=self.hyper_params,
            features=feats,
            targets=targs,
            save_path=self.it_dir,
            save_final_model=True,
            plot_feat_importance=True,
        )

        self.it_rf_model = rf[0]

        self.logger.info("Model trained successfully")

        self.logger.debug(f"Descriptor file list:\n{self.desc_flist}")
        self.logger.debug(f"Full file list:\n{self.full_flist}")


        # Setting up arguments for the _predict_for_files() function
        args = [
            (i, desc_file, full_file, model)
            for i, (desc_file, full_file) in enumerate(
                zip(self.desc_flist, self.full_flist)
            )
        ]

        self.logger.debug(f"Arguments for prediction processes:\n{args}")

        #Multiprocessing through all full & descriptor files to make predictions
        with Pool(self.n_cpus) as pool:
            pool.map(self._predict4Files, args)
            
        return self.it_rf_model

    def runIterations(self, held_out_test_feats: str, held_out_test_targs: str, held_out_test_full:str):
        """
        Description
        -----------
        Function to run the full RecDMTA workflow

        Parameters
        ---------
        held_out_test_desc (str)     Path to csv with descriptors for held out data to make predictions on
        held_out_test_targs (str)    Path to csv with descriptors for held out docking scores to compare against predictions

        Returns
        -------
        None
        """
        if self.n_cpus == -1:
            self.n_cpus = mp.cpu_count()

        self.logger.debug(f"Running with {self.n_cpus} CPUs")

        it_ran_ls = []

        # Starting iterations loop
        for self.iter in range(self.start_iter, self.start_iter + self.total_iters):

            # Checking to see if the iteration will run over the total runtime allocation
            if self.time_ran + (self.avg_runtime * 1.5) < self.max_it_runtime:
                it_start_time = time.time()

                self.logger.info(f"\n+===========Iteration: {self.iter}===========+\n")

                # Setting up the run directory, naming it _running/
                self.it_dir = self.run_dir / f"it{self.iter}_running"
                mk_it_dir = Path(self.it_dir)
                mk_it_dir.mkdir(exist_ok=True)

                # Setting up the training_data directory
                mk_train_data = mk_it_dir / "training_data"
                mk_train_data.mkdir(exist_ok=True)

                if self.iter - 1 != 0:
                    self.prev_it_dir = Path(self.run_dir) / f"it{self.iter - 1}"

                self.logger.debug(f"Compound selection arguments:\nIteration:{self.iter}\nPrevious Itetation Dir:{self.prev_it_dir}")
                self.selectCompounds(iteration=self.iter, prev_it_dir=self.prev_it_dir)

                self.runDocking()

                (
                    new_targs,
                    new_feats,
                ) = self.updateTrainingSet()

                self.retrainAndPredict(targs=new_targs, feats=new_feats)

                # Renaming iteration directory
                old_path = Path(self.it_dir)
                new_path = Path(self.run_dir) / f"it{self.iter}"
                old_path.rename(new_path)
                self.it_dir = new_path

                # Predict on held out test set
                cols = self.updated_feats.columns
                model = RFModel(docking_column=self.docking_column, logger=self.logger)

                # Read in target and feature data for held-out test set
                held_out_targ_df = pd.read_csv(held_out_test_targs, index_col="ID")
                held_out_targ_df = held_out_targ_df[held_out_targ_df[self.docking_column] != "False"]
                held_out_targ_df = held_out_targ_df[self.docking_column].astype(float)

                held_out_feat_df = pd.read_csv(held_out_test_feats, index_col="ID")[cols]
                held_out_feat_df = held_out_feat_df.loc[held_out_targ_df.index]

                # Run prediction using the RF wrapper
                pred_df = model.predict(
                    feats=held_out_feat_df,
                    save_preds=False,
                    final_rf=self.it_rf_model,
                    pred_col_name=f"pred_{self.docking_column}",
                    calc_mpo=True,
                    full_data_fpath=held_out_test_full,
                )
                
                self.logger.debug(f"Held-out predictions head:\n{pred_df.head()}")

                bias, sdep, mse, rmse, r2, pearson_r, pearson_p, true, pred = model._calculatePerformance(
                    feature_test=held_out_feat_df,
                    target_test=held_out_targ_df,
                    best_rf=self.it_rf_model,
                )
                stats = {
                    "Bias": round(bias, 4),
                    "SDEP": round(sdep, 4),
                    "MSE": round(mse, 4),
                    "RMSE": round(rmse, 4),
                    "r2": round(r2, 4),
                    "pearson_r": round(pearson_r, 4),
                    "pearson_p": round(pearson_p, 4),
                }

                with open(self.it_dir / "held_out_stats.json", "w") as f:
                    json.dump(stats, f, indent=4)

                pred_df.index.name = "ID"
                pred_df.to_csv(self.it_dir / "held_out_preds.csv")

                # Calculating timings
                it_fin_time = time.time()
                iter_time = it_fin_time - it_start_time
                self.run_times.append(iter_time)
                self.time_ran += iter_time
                it_ran_ls.append(self.iter)
                self.avg_runtime = np.mean(self.run_times)

                self.logger.info(f"+=========== Iteration Completed: {self.iter} ===========+")
                self.logger.info(f"Iteration Run Time: {round(iter_time, 1)}")
                self.logger.info(f"Average Iteration Run Time: {round(self.avg_runtime, 1)}")

        self.logger.info(f"Iterations Ran:\n{it_ran_ls}")
