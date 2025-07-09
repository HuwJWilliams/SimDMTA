import pandas as pd
import sys
from rdkit import Chem
import re
from pathlib import Path
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool
import os
import random as rand
import textwrap
from glob import glob
import numpy as np
import logging
import shutil
import portalocker
import filelock
import logging

# Import Openeye Modules
from openeye import oechem

# Muting GPU warning
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
from openeye import oequacpac, oeomega

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'


sys.path.insert(0, str(SCRIPTS_DIR / 'misc'))
from misc_functions import molid2BatchNo

sys.path.insert(0, str(SCRIPTS_DIR / 'dataset'))
from dataset_fns import DatasetAccessor

# Find Openeye licence
try:
    print("license file found: %s" % os.environ["OE_LICENSE"])
except KeyError:
    print("license file not found, please set $OE_LICENSE")
    sys.exit("Critical Error: license not found")

def updateDockCSV(dock_csv, ids, scores, scores_col: str = "Affinity(kcal/mol)", logger=None):
    max_retries = 10
    retry_delay = 10

    if logger is None:
        logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        try:
            with open(dock_csv, "r+") as f:
                portalocker.lock(f, portalocker.LOCK_EX)

                # Read inside the lock
                dock_df = pd.read_csv(f, index_col="ID", dtype=str)

                # Ensure scores list matches length of ids
                score_list = scores if isinstance(scores, list) else [scores] * len(ids)

                for id_, score in zip(ids, score_list):
                    if id_ in dock_df.index:
                        dock_df.at[id_, scores_col] = score

                # Write while still holding the lock
                f.seek(0)
                dock_df.to_csv(f)
                f.truncate()

                # Unlock happens when `with` block ends
                return

        except portalocker.exceptions.LockException:
            logger.warning(
                f"Lock attempt {attempt + 1}/{max_retries} failed — retrying in {retry_delay}s"
            )
            time.sleep(retry_delay)

def safeExtractTar(tar_file: Path, output_dir: Path, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)

    lock_path = tar_file.with_suffix(".lock")
    lock = filelock.FileLock(lock_path)

    try:
        with lock:
            if output_dir.exists():
                return True
            
            output_dir.mkdir(parents=True, exist_ok=True)

            command = ["tar", "-xzf", str(tar_file), "-C", str(output_dir)]
            subprocess.run(command, check=True)
            return True
    except Exception as e:
        logger.error(f"Failed to extract {tar_file}:\n{e}")

    finally:
        if lock_path.exists():
            lock_path.unlink

def wait4Docking(
    dock_csv: str,
    idxs_in_batch: list,
    scores_col: str,
    check_interval: int,
    ascending: bool = True,
    logger=None
):
    """
    Description
    -----------
    Function to check whether or not the docking of provided molecules has completed.
    
    Parameters
    ----------
    dock_csv (str)              Path to docking CSV file, where docking scores are held.
    idxs_in_batch (list)         List of molecule IDs in the dock_csv file which are being docked.
    scores_col (str)            Name of the column the docking scores are being saved in the data frame.
    check_interval (int)        How often (in seconds) to check if docking has finished.
    ascending (bool)            Sort order for the docking score (if, e.g., lower is better, use True).
    
    Returns
    -------
    None
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Loop until there are no pending molecules
    while True:
        # Read the docking CSV file
        try:
            dock_df = pd.read_csv(dock_csv, index_col="ID", dtype=str)
        except Exception as e:
            logger.error(f"Error reading {dock_csv}: {e}")
            time.sleep(check_interval)
            continue

        # Filter for the current batch IDs
        df_with_idx = dock_df[dock_df.index.isin(idxs_in_batch)]
        pending_docking = df_with_idx[df_with_idx[scores_col] == "PD"]

        if pending_docking.empty:
            logger.info("All docking scores present.")
            break

        logger.info(f"Waiting for the following molecules to dock: {list(pending_docking.index)}")

        ids_changed = []
        for ids in pending_docking.index:
            tar_file = Path(PROJ_DIR) / "docking" / "PyMolGen" / f"{ids}.tar.gz"

            if tar_file.exists():

                tar_file = Path(PROJ_DIR) / "docking" / "PyMolGen" / f"{ids}.tar.gz"
                output_dir = Path(PROJ_DIR) / "docking" / "PyMolGen" / f"extracted_{ids}"

                if safeExtractTar(tar_file=tar_file, output_dir=output_dir, logger=logger):
                    # Unzip the .csv.gz file
                    gz_file = output_dir / f"{ids}" / f"{ids}_all_scores.csv.gz"

                    try:
                        id_dock_scores = pd.read_csv(str(gz_file), index_col="ID").sort_values(
                            ascending=ascending, by=scores_col
                        )
                        dock_score = id_dock_scores[scores_col].iloc[0]
                        
                        updateDockCSV(
                            dock_csv=dock_csv, 
                            ids=[ids], 
                            scores=dock_score, 
                            scores_col=scores_col, 
                            logger=logger
                        )

                        shutil.rmtree(output_dir, ignore_errors=True)
                        ids_changed.append(ids)

                    except Exception as e:
                        logger.error(f"Failed to process extracted data for {ids}. Error: {e}")
                        
                        # ✅ Reset PD to blank after failed processing
                        updateDockCSV(
                            dock_csv=dock_csv,
                            ids=[ids],
                            scores="",  # or "False"
                            scores_col=scores_col,
                            logger=logger
                        )
                    logger.error(f"Failed to process extracted data for {ids}. Error: {e}")

        if ids_changed:
            pending_docking = pending_docking[~pending_docking.index.isin(ids_changed)]
            logger.debug(f"Processed IDs removed from pending docking: {list(ids_changed)}")

        time.sleep(check_interval)

def getUndocked(dock_df: pd.DataFrame, idxs_in_batch: list, scores_col: str, logger=None):
    """
    Description
    -----------
    Function to obtain which of the molecule IDs provided have not alreeady been docked

    Parameters
    ----------
    dock_df (pd.DataFrame)      Data frame containing docking data
    idx_in_batch (list)         List of molecule IDs in the dock_df which are being docked
    scores_col (str)            Name of the column the docking scores are being saved in the data frame
    
    Returns
    -------
    List of undocked molecule
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    df = dock_df.loc[idxs_in_batch]
    df[scores_col] = df[scores_col].astype(str).str.strip().replace(["PD", "False", ""], np.nan)
    df[scores_col] = pd.to_numeric(df[scores_col], errors="coerce")    

    undocked = df[df[scores_col].isna()]
    docked = df.dropna(subset=[scores_col])

    logger.debug(f"Found {len(undocked)} undocked molecules: {undocked}.")
    for mol_id, score in docked[scores_col].items():
        logger.debug(f"Molecule {mol_id} already docked with score: {score}")

    logger.debug(f"Filtered docking DataFrame:\n{df[[scores_col]]}")

    return undocked

def cleanFiles(fpath:str=f"{str(PROJ_DIR)}/datasets/PyMolGen/docking/",
                      fname:str="PMG_docking_*.csv",
                      contaminants: list=["PD"],
                      replacement: str="",
                      index_col:str='ID',
                      logger=None):
    """
    Description
    -----------
    Function to remove any contaminants from files (e.g., 'PD', 'NaN', False, ...)
    
    Parameters
    ----------
    fpath (str)             Path to docking files
    fname (str)             Docking file file name. Can be either generic (e.g., * or ?) or specific
    docking column (str)    Column which to remove contaminants from
    contaminants (list)     Values to remove
    replacement (str)       Values to replace contaminants with
    index_col (str)         Name of index column

    Returns
    -------
    None
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    working_path = fpath + fname
    replace_dict = {value: replacement for value in contaminants}

    if "*" in fname or "?" in fname:
        fpath_ls = glob(working_path)

    else:
        fpath_ls = [working_path]

    for path in fpath_ls:
        working_df = pd.read_csv(path, index_col=index_col)

        contaminant_counts = {contaminant: 0 for contaminant in contaminants}

        for contaminant in contaminants:
            if contaminant == "NaN":
                contaminant_counts["NaN"] = working_df.isna().sum().sum()
            else:
                contaminant_counts[contaminant] = (working_df == contaminant).sum().sum()

        working_df.replace(replace_dict, inplace=True)
        if "NaN" in contaminants or np.nan in contaminants:
            working_df.fillna(replacement, inplace=True)

        working_df.to_csv(path, index_label=index_col)

        for contaminant, count in contaminant_counts.items():
            logger.info(f" - Found and replaced {count} instances of '{contaminant}'.\n")
        

class RunGNINA:
    """
    Description
    -----------
    Class to do all things GNINA docking
    """

    def __init__(
        self,
        docking_dir: str,
        molid_ls: list,
        smi_ls: list,
        receptor_path: str,
        max_confs: int = 1000,
        center_x: float = 14.66,
        center_y: float = 3.41,
        center_z: float = 10.47,
        size_x: float = 17.67,
        size_y: float = 17.00,
        size_z: float = 13.67,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        cpu: int = 1,
        addH: int = 0,
        stripH: int = 1,
        seed: int = rand.randint(0, 2**31),
        cnn_scoring: str = "rescore",
        gnina_path: str = f"{str(PROJ_DIR)}/scripts/docking/gnina",
        env_name: str = "phd_env",
        use_autobox: bool=False,
        autobox_ligand_pdb: str=None,
        docking_general_file: str=f"{PROJ_DIR}/datasets/PyMolGen/docking/PMG_docking_*.csv",
        logger=None,
        log_path=None,
        log_level: str="DEBUG"
    ):
        """
        Description
        -----------
        Initialisation of the class, setting all of the GNINA values required for the docking.
        Makes directories for each molecule ID.

        Parameters
        ----------
        docking_dir (str)           Directory where all of the docking will take place (particularly saving the results to)
        molid_ls (list)             List of molecule ids which will be used to save the data under
        smi_ls (list)               List of SMILES strings which you wish to find the docking score of
        receptor_path (str)         Pathway to the receptor
        center (x,y,z) (float)      X, Y, Z coordinate of the center of the focussed box
        size (x, y, z) (float)      Size of the focussed box in the X, Y, Z dimensions
        exhaustiveness (int)        Exhaustiveness of the global search
        num_modes (t1_1_rint)       Number of binding modes to generate
        cpu (int)                   Number of CPUs to use (default keep as 1)
        addH (int)                  Automatically adds hydrogens in ligands (0= off, 1= on)
        seed (int)                  Setting the random seed
        cnn_scoring (str)           Determines at what point the CNN scoring function is used:
                                    none = No CNNs used for dockingm uses empirical scoring function
                                    rescore = CNN used for reraning of final poses (least compuationally expensive CNN option)
                                    refinement = CNN to refine poses after Monte Carlo chains (10x slower than rescore)
                                    all = CNN used as scoring dunction throughout (extremely computationally expensive)
        gnina_path (str)            Path to the gnina executable file
        env_name (str)              Name of conda environment to use

        Returns
        -------
        Initialised class
        """

        if logger:
            # Use provided logger (e.g., from SimDMTA)
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

            self.logger.info(f"Initialised logger: {log_path}")


        self.logger.debug(f"Random Seed: {seed}")

        self.docking_dir = Path(docking_dir)  # Ensure it's a Path object

        self.tar_mol_dir_path_ls = [
            str(self.docking_dir / f"{molid}.tar.gz") for molid in molid_ls
        ]
        self.mol_dir_path_ls = [
            str(self.docking_dir / molid / "") for molid in molid_ls  # trailing slash if needed
        ]

        for dir, tar_dir in zip(self.mol_dir_path_ls, self.mol_dir_path_ls):
            if Path(dir).exists() or Path(tar_dir).exists():
                self.logger.warning(f"{dir} exists")
            else:
                Path(dir).mkdir()

        # Instantiating docking variables
        self.targ_file = docking_general_file
        self.docking_dir = docking_dir
        self.smi_ls = smi_ls
        self.molid_ls = molid_ls
        self.receptor_path = receptor_path
        self.max_confs = max_confs
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.num_cpu = cpu
        self.addH = addH
        self.stripH = stripH
        self.seed = seed
        self.cnn_scoring = cnn_scoring
        self.gnina_path = gnina_path
        self.env_name = env_name
        self.use_autobox = use_autobox
        self.autobox_ligand_pdb = autobox_ligand_pdb

    def _smiles2SDF(self, smi: str, sdf_fpath: str):
        """
        Description
        -----------
        Function to make .sdf file from SMILES string

        Parameters
        ----------
        smi (str)       SMILES string of molecule to enhance
        sdf_fpath (str) File path/name to save the .sdf under

        """

        if not isinstance(smi, str):
            smi= str(smi)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            self.logger.debug(smi)
            self.logger.debug(sdf_fpath)

        h_mol = Chem.AddHs(mol)

        with open(sdf_fpath, "w") as file:
            file.write(Chem.MolToMolBlock(h_mol))

    def _molPrep(self, smi: str, molid: str, mol_dir: str):
        """
        Description
        -----------
        Prepare molecule by generating 3D coordinates, calculating ionisation state
        at pH 7.4 and saving an image of the molecule.
            
        Parameters
        ----------
        smi (str)           SMILES string of molecule to set to pH 7.4
        molid (str)         ID of molecule to save .sdf file as
        mol_dir (str)       Path to directory containing the molecule (set in initialisation)

        Returns
        -------
        Pathway to converted .sdf file
        """

        lig_sdf_path = Path(mol_dir) / f"{molid}.sdf"
        pH74_path = Path(mol_dir) / f"{molid}_pH74.sdf"

        self._smiles2SDF(smi, str(lig_sdf_path))

        try:
            ifs = oechem.oemolistream()
            ifs.open(str(lig_sdf_path))

            ofs = oechem.oemolostream()
            ofs.SetFormat(oechem.OEFormat_SDF)
            ofs.open(str(pH74_path))

            mol = oechem.OEGraphMol()

            while oechem.OEReadMolecule(ifs, mol):
                if oequacpac.OESetNeutralpHModel(mol):
                    oechem.OEAddExplicitHydrogens(mol)
                    oechem.OEWriteMolecule(ofs, mol)
            ifs.close()
            ofs.close()

            return str(pH74_path)

        except Exception as e:
            self.logger.error(f"Failed to convert mol to pH 7.4 for the following reason:\n{e}")
            return None

    def _phAdjust(self):
        """
        Description
        -----------
        Wrapper function to carry out _molPrep() on self.sdf_path_ls

        Parameters
        ----------
        None

        Returns
        -------
        List of pH 7.4 .sdf file paths
        """

        self.sdf_path_ls = []
        for molid, smi, molid_dir in zip(
            self.molid_ls, self.smi_ls, self.mol_dir_path_ls
        ):
            mol_dir_path = Path(molid_dir)

            if mol_dir_path.suffixes == [".tar", ".gz"]:
                subprocess.run(["tar", "-xzf", molid_dir], check=True)
                path = self._molPrep(smi=smi, molid=molid, mol_dir=molid_dir)
                self.sdf_path_ls.append(str(path))
            else:
                path = self._molPrep(smi=smi, molid=molid, mol_dir=molid_dir)
                self.sdf_path_ls.append(str(path))

        return self.sdf_path_ls

    def disallowedFragTautomers(self, molecule):
        """
        Description
        -----------
        Function to remove tautomeric fragments which are disallowed
        
        Parameters
        ----------
        molecule (object)       Molecule object
        
        Returns
        -------
        Either True or False based on whether the molecule has the fragment or not
        """

        fragment_list = ["N=CO", "C=NCO", "C(O)=NC", "c=N", "N=c", "C=n", "n=C"]

        for fragment in fragment_list:
            fragment_search = oechem.OESubSearch(fragment)
            oechem.OEPrepareSearch(molecule, fragment_search)
            if fragment_search.SingleMatch(molecule):
                return False
                break

        return True

    def generateConformers(self, sdf_fpath: str):
        """
        Description
        -----------
        Function to search through conformational space of a molecule and save each conformer to a 
        master .sdf file

        Parameters
        ----------
        sdf_fpath (str)     Path of the original .sdf file for a given molecule

        Returns
        -------
        File path of the sdf file containing all of the conformers made
        """
        lig_in_fname = Path(sdf_fpath).name
        lig_in_fpath = Path(sdf_fpath).parent
        lig_out_fpath = str(lig_in_fpath) + "/all_confs_" + str(lig_in_fname)
        ligand_prefix = str(lig_in_fname)[:-4]

        omega_opts = oeomega.OEOmegaOptions()
        omega_opts.SetEnergyWindow(20)
        omega_opts.SetMaxSearchTime(600)
        omega_opts.SetSearchForceField(7)
        omega_opts.SetRMSThreshold(0.5)
        omega_opts.SetMaxConfs(self.max_confs)
        omega_opts.SetTorsionDrive(True)
        omega_opts.SetStrictStereo(False)

        omega = oeomega.OEOmega(omega_opts)

        ifs = oechem.oemolistream()
        ifs.open(sdf_fpath)

        ofs_sdf = oechem.oemolostream()
        ofs_sdf.SetFormat(oechem.OEFormat_SDF)
        ofs_sdf.open(lig_out_fpath)

        # Output Conformers in a mol2 file for showing in VMD

        ofs_mol2 = oechem.oemolostream()
        ofs_mol2.SetFormat(oechem.OEFormat_MOL2)
        ofs_mol2.open(lig_out_fpath[:-4] + ".mol2")

        conf_isomer_ids = open(lig_out_fpath[:-4] + "_conf_isomers.dat", "w")
        conf_isomer_ids.write("conf_n,tauto,enant\n")

        tautomer_opts = oequacpac.OETautomerOptions()
        tautomer_opts.SetMaxSearchTime(300)
        tautomer_opts.SetRankTautomers(True)
        tautomer_opts.SetCarbonHybridization(False)

        # enantiomer options
        flipper_opts = oeomega.OEFlipperOptions()
        flipper_opts.SetMaxCenters(12)
        flipper_opts.SetEnumSpecifiedStereo(
            False
        )  # Changed to False to preserve defined stereochemistry
        flipper_opts.SetEnumNitrogen(True)
        flipper_opts.SetWarts(False)

        # generate tautomers, enantiomers and conformers
        # Record number of tautomers, enantiomers, disallowed isomers
        n_tauto = 0
        n_enant = []
        n_confs = []
        n_disallowed = 0
        conf_i = 1

        for mol in ifs.GetOEMols():
            for tautomer in oequacpac.OEEnumerateTautomers(mol, tautomer_opts):
                n_tauto += 1
                n_enant.append(0)
                n_confs.append([])

                for enantiomer in oeomega.OEFlipper(tautomer, flipper_opts):
                    n_enant[-1] += 1
                    ligand = oechem.OEMol(enantiomer)
                    ligand.SetTitle(ligand_prefix)

                if self.disallowedFragTautomers(ligand):
                    ret_code = omega.Build(ligand)
                    if ret_code == oeomega.OEOmegaReturnCode_Success:
                        n_confs[-1].append(ligand.NumConfs())
                        # Add SD data to indicate tautomer/enantiomer number:
                        oechem.OESetSDData(ligand, "tautomer_n", str(n_tauto))
                        oechem.OESetSDData(ligand, "isomer_n", str(n_enant[-1]))
                        oechem.OEWriteMolecule(ofs_sdf, ligand)
                        oechem.OEWriteMolecule(ofs_mol2, ligand)
                        conf_i += 1
                else:
                    n_disallowed += 1
        conf_isomer_ids.close()

        with open(str(lig_in_fpath) + "/conf_generation.log", "w") as log:
            log.write(f"Conformer generation: tautomers: {n_tauto}\n")
            log.write(f"                      enantiomers: {n_enant}\n")
            log.write(f"                      number disallowed: {n_disallowed}\n")
            log.write(
                f"                      final number: {sum(n_enant) - n_disallowed}\n"
            )
            log.write(
                f"                      number of individual 3D conformers: {n_confs}\n"
            )

        self.logger.debug(
            f"Conformers generated: {n_confs} |"
            f"Tautomers: {n_tauto} |"
            f"Enantiomers: {n_enant} |"
            f"Disallowed: {n_disallowed}"
        )

        ifs.close()
        ofs_sdf.close()
        ofs_mol2.close()

        return str(lig_out_fpath)

    def _processMolWrapper(self, molid_dir: str, smi: str, molid: str):
        """
        Desctiption
        -----------
        Function to wrap the processing of molecules (molecule preparation and conformer search) so that it can
        be used in multiprocessing

        Parameters
        ----------
        molid_dir (str)     Pathway to directory containing molecule information
        smi (str)           SMILES string of the molecule to process
        molid (str)         ID of molecule to process

        Returns
        -------
        Pathway to .sdf file containing all conformations generated for the given SMILES
        """
        mol_dir_path = Path(molid_dir)

        if mol_dir_path.suffixes == [".tar", ".gz"]:
            subprocess.run(["tar", "-xzf", molid_dir], check=True)

        non_conf_path = self._molPrep(smi=smi, molid=molid, mol_dir=molid_dir)
        conf_path = self.generateConformers(non_conf_path)

        return conf_path

    def processMols(self, use_multiprocessing):

        self.sdf_path_ls = []
        if use_multiprocessing:
            with Pool() as pool:
                results = pool.starmap(
                    self._processMolWrapper,
                    zip(self.mol_dir_path_ls, self.smi_ls, self.molid_ls),
                )
            self.sdf_path_ls.extend(results)

        else:
            for molid_dir, smi, molid in zip(
                self.mol_dir_path_ls, self.smi_ls, self.molid_ls
            ):
                self.sdf_path_ls.append(
                    self._processMolWrapper(molid_dir, smi, molid)
                )

        return self.sdf_path_ls

    def _makeSDFs(self):
        """
        Description
        -----------
        Simplified SMILES to .sdf conversion without pH considerations

        Parameters
        ----------
        None

        Returns
        -------
        List of .sdf file paths
        """
        self.sdf_path_ls = []

        for molid, smi, molid_dir in zip(
            self.molid_ls, self.smi_ls, self.mol_dir_path_ls
        ):

            # Convert SMILES to RDKit Mol
            mol = Chem.MolFromSmiles(smi)

            if Path(molid_dir + molid + ".sdf").exists():
                self.logger.warning(f"{molid}.sdf file exists")

            if mol is not None:
                # Write to SDF file
                with Chem.SDWriter(molid_dir + molid + ".sdf") as writer:
                    writer.write(mol)
                self.sdf_path_ls.append(molid_dir + molid + ".sdf")
            else:
                self.logger.error(f"Invalid SMILES string: \n{smi}")

        return self.sdf_path_ls

    def createSubmissionScript(
        self,
        molid: str,
        run_hrs: int,
        mol_dir: str,
        sdf_filename: str,
        output_filename: str,
        log_filename: str,
        run_mins: int = 0,
        template: str = "slurm",
        custom_template: str = None
    ):
        """
        Create a GNINA job submission shell script.

        Returns
        -------
        Path to the created shell script, and molecule directory path.
        """

        gnina_parameters = {
            "molid": molid,
            "run_hrs": run_hrs,
            "run_mins": run_mins,
            "mol_dir": mol_dir,
            "sdf_filename": sdf_filename,
            "output_filename": output_filename,
            "log_filename": log_filename,
            "receptor_path": self.receptor_path,
            "gnina_path": self.gnina_path,
            "env_name": self.env_name,
            "autobox_ligand_pdb": getattr(self, "autobox_ligand_pdb", ""),
            "num_cpu": self.num_cpu,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
            "exhaustiveness": self.exhaustiveness,
            "num_modes": self.num_modes,
            "addH": self.addH,
            "stripH": self.stripH,
            "seed": self.seed,
            "cnn_scoring": self.cnn_scoring,
        }

        # Custom template override
        if custom_template:
            script_template = custom_template

        elif template == "local":
            script_template = """\
                #!/bin/bash
                source activate {env_name}

                {gnina_path} \\
                    --receptor "{receptor_path}" \\
                    --ligand "{mol_dir}{sdf_filename}" \\
                    --out "{mol_dir}{output_filename}" \\
                    --log "{mol_dir}{log_filename}" \\
                    --center_x {center_x} --center_y {center_y} --center_z {center_z} \\
                    --size_x {size_x} --size_y {size_y} --size_z {size_z} \\
                    --exhaustiveness {exhaustiveness} --num_modes {num_modes} \\
                    --cpu {num_cpu} --no_gpu --addH {addH} --stripH {stripH} \\
                    --seed {seed} --cnn_scoring "{cnn_scoring}"
            """

        else:
            # Conditional ligand/autobox block
            if self.use_autobox:
                docking_args = '--autobox_ligand "{autobox_ligand_pdb}"'
            else:
                docking_args = """\
                    --ligand "{mol_dir}/{sdf_filename}" \\
                    --center_x {center_x} --center_y {center_y} --center_z {center_z} \\
                    --size_x {size_x} --size_y {size_y} --size_z {size_z}"""

            script_template = f"""\
                #!/bin/bash
                #SBATCH --export=ALL
                #SBATCH --time {{run_hrs}}:{{run_mins}}:00
                #SBATCH --job-name=dock_{{molid}}
                #SBATCH --ntasks={{num_cpu}}
                #SBATCH --partition=standard
                #SBATCH --account=palmer-addnm
                #SBATCH --output={{mol_dir}}/slurm-%j.out

                set -e
                cd {{mol_dir}}
                echo "Running in $(pwd)"
                ls -la

                if [ -f /opt/software/scripts/job_prologue.sh ]; then
                    /opt/software/scripts/job_prologue.sh
                fi

                module purge
                module load anaconda/python-3.9.7
                source activate {{env_name}}

                {{gnina_path}} \\
                    --receptor "{{receptor_path}}" \\
                    {docking_args} \\
                    --out "{{output_filename}}" \\
                    --log "{{log_filename}}" \\
                    --exhaustiveness {{exhaustiveness}} --num_modes {{num_modes}} \\
                    --cpu {{num_cpu}} --no_gpu --addH {{addH}} --stripH {{stripH}} \\
                    --seed {{seed}} --cnn_scoring "{{cnn_scoring}}"

                if [ -f /opt/software/scripts/job_epilogue.sh ]; then
                    /opt/software/scripts/job_epilogue.sh
                fi
            """
        script_template = textwrap.dedent(script_template)
        script_text = script_template.format(**gnina_parameters)
        script_path = Path(mol_dir) / f"{molid}_docking_script.sh"

        with open(script_path, "w") as file:
            file.write(script_text)

        subprocess.run(["chmod", "+x", str(script_path)], check=True)

        self.logger.debug(
            f"Created docking script at {script_path} with parameters:\n{gnina_parameters}"
        )


        return script_path, mol_dir

    def submitDockingScript(self, docking_script_fpath: str, mol_dir: str):
        try:
            result = subprocess.run(
                ["sbatch", docking_script_fpath],
                capture_output=True,
                text=True,
                check=True,
            )

            stdout_path = Path(mol_dir) / "stdout.txt"
            stderr_path = Path(mol_dir) / "stderr.txt"

            with open(stdout_path, "w") as stdout_file:
                stdout_file.write(result.stdout)

            with open(stderr_path, "w") as stderr_file:
                stderr_file.write(result.stderr)

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                jobid = match.group(1)
                self.logger.debug(f"Docking submitted. Job ID: {jobid}")
                return jobid
            else:
                self.logger.error("Failed to extract job ID from sbatch output.")
                return None

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in submitting job: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
        
    def submitMultipleJobs(self, run_hrs: int, run_mins: int = 0, use_multiprocessing=True):
        """
        Description
        -----------
        Function to submit numerous docking jobs in parallel.

        Parameters
        ----------
        run_hrs (int)       Maximum time to run the docking for
        username (str)      Username which the submitted scripts will be under
        run_mins (int)      Minutes runtime to run docking
        """

        if use_multiprocessing:
            with Pool() as pool:
                results = pool.starmap(
                    self.createSubmissionScript,
                    [
                        (
                            molid,
                            run_hrs,
                            mol_dir,
                            f"all_confs_{molid}_pH74.sdf",
                            f"{molid}_pose.sdf",
                            f"{molid}.log",
                            run_mins,
                        )
                        for molid, mol_dir in zip(self.molid_ls, self.mol_dir_path_ls)
                    ],
                )

            shell_scripts = [result[0] for result in results]
            mol_dirs = [result[1] for result in results]

            with Pool() as pool:
                results = pool.starmap(
                    self.submitDockingScript,
                    [
                        (script, mol_dir)
                        for script, mol_dir in zip(shell_scripts, mol_dirs)
                    ],
                )

        job_ids = [jobid for jobid in results if jobid is not None]

        return job_ids

    def makeDockingCSV(
        self,
        save_data: bool = True,
        mol_dir_path_ls: list = None,
        molid_ls: list = None,
    ):
        """
        Parse GNINA .log files to extract docking scores and update docking CSV files.
        Skips molecules if a valid docking score is already present.
        """
        if mol_dir_path_ls is not None and molid_ls is not None:
            self.mol_dir_path_ls = mol_dir_path_ls
            self.molid_ls = molid_ls

        top_cnn_aff_ls = []
        top_aff_ls = []

        for mol_dir, molid in zip(self.mol_dir_path_ls, self.molid_ls):
            combined_df = pd.DataFrame()
            mol_dir_path = Path(mol_dir)
            log_file_path = mol_dir_path / f"{molid}.log"
            tar_file_path = mol_dir_path.with_suffix(".tar.gz")

                # --- Check for existing docking score ---
            try:
                batch_no = molid2BatchNo(molid=molid, prefix="PMG-", dataset_file=self.targ_file)
                dock_batch_csv = self.targ_file.replace("*", str(batch_no))
                docking_df = pd.read_csv(dock_batch_csv, index_col="ID", dtype=str)

                if self._isDocked(molid, docking_df):
                    self.logger.info(f"Skipping {molid} - already docked")
                    top_aff_ls.append(docking_df.at[molid, "Affinity(kcal/mol)"])
                    top_cnn_aff_ls.append("Skipped")
                    continue

            except Exception as e:
                self.logger.warning(f"Could not verify existing docking score for {molid}: {e}")
                try:
                    docking_df.at[molid, "Affinity(kcal/mol)"] = ""
                    docking_df.to_csv(dock_batch_csv, index_label="ID")
                except Exception as ex:
                    self.logger.error(f"Failed to clear PD for {molid}: {ex}")

            extracted = False
            if not log_file_path.exists() and tar_file_path.exists():
                subprocess.run(["tar", "xzf", str(tar_file_path), "-C", str(mol_dir_path.parent)])
                extracted = True

            try:
                with log_file_path.open("r") as file:
                    lines = file.readlines()
            except Exception as e:
                self.logger.error(f"Cannot read log file for {molid}: {e}")
                continue

            table_start_indices = [i for i, line in enumerate(lines) if line.startswith("mode |")]
            self.logger.debug(f"Tables start on lines: {table_start_indices}")

            if not table_start_indices:
                combined_df = pd.DataFrame(
                    data={ "ID": [f"{molid}_conf_0_pose_0"], "conf_no": [0], "Pose_no": [0],
                        "Affinity(kcal/mol)": ["False"], "Intramol(kcal/mol)": ["False"],
                        "CNN_Pose_Score": ["False"], "CNN_affinity": ["False"] })

            for j, start_idx in enumerate(table_start_indices):
                df_lines = lines[start_idx + 3:]
                pose_ls, aff_ls, intra_ls, cnn_pose_score_ls, cnn_aff_ls = [], [], [], [], []

                for line in df_lines:
                    if not line.strip() or line.startswith("mode |") or line.startswith("Using random seed"):
                        break
                    items = re.split(r"\s+", line.strip())
                    pose_ls.append(items[0])
                    aff_ls.append(items[1])
                    intra_ls.append(items[2])
                    cnn_pose_score_ls.append(items[3])
                    cnn_aff_ls.append(items[4])

                pose_df = pd.DataFrame({
                    "ID": [f"{molid}_conf_{j}_pose_{pose}" for pose in pose_ls],
                    "conf_no": j,
                    "Pose_no": pose_ls,
                    "Affinity(kcal/mol)": aff_ls,
                    "Intramol(kcal/mol)": intra_ls,
                    "CNN_Pose_Score": cnn_pose_score_ls,
                    "CNN_affinity": cnn_aff_ls,
                })

                combined_df = pd.concat([combined_df, pose_df], ignore_index=True)

            try:
                max_cnn = combined_df["CNN_affinity"].astype(float).max()
                min_aff = combined_df["Affinity(kcal/mol)"].astype(float).min()
            except:
                max_cnn = "False"
                min_aff = "False"

            top_cnn_aff_ls.append(max_cnn)
            top_aff_ls.append(min_aff)

            if save_data:
                self.logger.info("Saving CSV...")
                output_path = Path(mol_dir) / f"{molid}_all_scores.csv.gz"
                combined_df.set_index("ID").to_csv(output_path, compression="gzip", index_label="ID")

                try:
                    docking_df.at[molid, "Affinity(kcal/mol)"] = min_aff
                    docking_df.to_csv(dock_batch_csv, index_label="ID")
                except Exception as e:
                    self.logger.error(f"Could not update docking CSV for {molid}: {e}")

            if extracted and mol_dir_path.exists():
                shutil.rmtree(mol_dir_path, ignore_errors=True)

        return self.molid_ls, top_cnn_aff_ls, top_aff_ls

    def compressDockingFiles(self):
        """
        Description
        -----------
        Function to compress each of the created molecule directory to save data

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for mol_dir in self.mol_dir_path_ls:
            # Removing the '/'
            mol_dir_path = Path(mol_dir)
            mol_dir_name = str(mol_dir_path.name)

            if not mol_dir_path.exists():
                self.logger.warning(f"Warning: Directory {mol_dir_path} does not exist. Skipping compression.")
                continue
                
            try:
                subprocess.run(
                    [
                        "tar",
                        "-czf",
                        f"{mol_dir_name}.tar.gz",
                        "--remove-files",
                        mol_dir_name,
                    ],
                    cwd=self.docking_dir,
                    check=True,
                )
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error compressing {mol_dir_path}:\n{e}")

    def _isDocked(self, molid, docking_df):
        try:
            score = docking_df.at[molid, "Affinity(kcal/mol)"]
            return pd.notna(score) and str(score).lower() not in ["", "false", "pd"]
        except KeyError:
            return False