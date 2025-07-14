import pandas as pd
import numpy as np
from glob import glob
import re
import fcntl
import time
import subprocess
from pathlib import Path
import json
from rdkit import Chem
import shutil
from PIL import Image, ImageFilter
import imageio
import logging

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
DATASETS_DIR = PROJ_DIR / 'datasets'

def wait4Jobs(job_id_ls: list, username: str = "yhb18174", wait_time: int = 60, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)

    while True:
        dock_jobs = []
        squeue = subprocess.check_output(["squeue", "--users", username], text=True)
        lines = squeue.splitlines()

        job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0}

        # Find which submitted job IDs are still in the squeue output
        dock_jobs = set(job_id for job_id in job_id_ls if job_id in job_lines)

        if not dock_jobs:
            return False

        if len(dock_jobs) < 10:
            job_message = (
                f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            )
        else:
            job_message = f"Waiting for {len(dock_jobs)} jobs to finish"

        logger.info(f"{job_message.ljust(80)}")
        time.sleep(wait_time)


def molid2BatchNo(molid: str, prefix: str, dataset_file: str, chunksize: int = 100000, file_ls: list[str] = None, logger=None):
    """
    Get the batch number a molecule belongs to based on its ID.

    Parameters
    ----------
    molid : str
        Molecule ID (e.g., "PMG-12345")
    prefix : str
        ID prefix (e.g., "PMG-")
    dataset_file : str
        Pattern for glob, e.g., 'data/full_data/PMG_rdkit_*.csv.gz'
    chunksize : int
        Molecules per batch
    file_ls : list[str], optional
        Precomputed, ordered list of file paths (to avoid globbing every time)

    Returns
    -------
    int
        Batch number the molecule belongs to
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.debug(f"Batch dataset files: {dataset_file}")

    mol_no = int(molid.replace(prefix, ""))

    logger.debug(f"Stripped molid {molid} of prefix '{prefix}' leaving {mol_no}")

    if file_ls is None:
        file_ls = glob(str(dataset_file))
        file_ls.sort(key=lambda x: int(re.search(r"\d+", x).group()))
    else:
        if "*" in dataset_file:
            file_ls = glob(dataset_file)
        else:
            file_ls = file_ls

    logger.debug(f"File list: {file_ls}")

    batch_number = (mol_no - 1) // chunksize + 1

    if batch_number > len(file_ls):
        raise ValueError(f"Batch number {batch_number} exceeds available dataset files.")

    return batch_number


def lockFile(file_path: str, logger=None):
    """
    Description
    -----------
    Function to lock a file to gain exclusive access to the file. Waits if file is locked

    Parameters
    ----------
    path (str)      Path to file
    filename (str)  File to lock

    Returns
    -------
    Locked file
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    while True:
        try:
            with open(file_path, "r", newline="") as file:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug(f"Acquired lock on {file}")
                return file
        except BlockingIOError:
            logger.debug(f"File {file} is locked. Waiting...")
            time.sleep(30)


def unlockFile(file: object):
    """
    Description
    -----------
    Function to unlock file locked from lockFile function

    Parameters
    ----------
    file (object)       File object to unlock

    Returns
    -------
    Unlocked file

    """
    return fcntl.flock(file, fcntl.LOCK_UN)


def csv2JSON(experiment_ls: list, results_dir: str):
    results_dir = Path(results_dir)
    if not results_dir.name.endswith("/"):
        results_dir = results_dir / ""

    for x in experiment_ls:
        experiment_dir = results_dir / x
        for n in range(countIterations(experiment_dir)):
            held_out_dir = experiment_dir / f"it{n+1}" / "held_out_test"
            file_path = Path(held_out_dir) / "held_out_test_stats.csv"
            if file_path.is_file():
                df = pd.read_csv(file_path, index_col="Unnamed: 0")
                stats_dict = df["Value"].to_dict()
                with open(f"{file_path.parent}/held_out_stats.json", "w") as f:

                    json.dump(stats_dict, f, indent=4)
            else:
                continue


def countIterations(
    results_dir: str,
    logger=None
):
    """
    Description
    -----------
    Function to count the number of completed iterations carried out in a given results directory

    Parameters
    ----------
    None

    Returns
    -------
    Number of completed iterations (doesnt count the ones suffixed with '_running')
    """
    results_dir = Path(results_dir)

    # List all directories starting with 'it'
    it_dirs = [
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("it")
    ]

    # Filter out directories ending with '_running'
    completed_it_dirs = [d for d in it_dirs if not d.name.endswith("_running")]

    num_iters = len(completed_it_dirs) - 1

    return num_iters


def countConformations(sdf_file: str):
    """
    Description
    -----------
    Function to count the number of conformations in an .sdf file

    Parameters
    ----------
    sdf_file (str)      SDF fpath

    Returns
    -------
    Number of conformations/molecules in an sdf file
    """

    # Read the SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    # Initialize counter for conformations
    conf_count = 0

    for mol in supplier:
        if mol is not None:
            # Get the number of conformations for each molecule
            conf_count += mol.GetNumConformers()

    return conf_count


def getMolsBetweenIters(experiment_dir: str, start_iter: int, end_iter: int):
    chosen_mols = pd.read_csv(experiment_dir + "/chosen_mol.csv", index_col=False)
    molid_ls = []

    for _, rows in chosen_mols.iterrows():
        if start_iter <= int(rows["Iteration"]) <= end_iter:
            molid_ls.append(rows["ID"])

    return molid_ls


def molid2Smiles(molid: str, prefix: str, data_fpath: str, chunksize: int = 100000):
    batch = molid2BatchNo(
        molid=molid, prefix=prefix, dataset_file=data_fpath, chunksize=chunksize
    )

    data_fpath = data_fpath.replace("*", str(batch))
    batch_df = pd.read_csv(data_fpath, index_col="ID")
    smi = batch_df.loc[molid, "Kekulised_SMILES"]

    return smi


def molidList2Smiles(
    molids: list, prefix: str, data_fpath: str, chunksize: int = 100000,
):
    
    """
    Description
    -----------
    Function to get the smiles for the molecule IDs provided

    Parameters
    ----------
    molids (str)        ID list of molecule
    prefix (str)        Prefix of the molecule ID == 'PMG-'
    data_fpath (str)    Common filename of dataset
    chunksize (int)     Number of molecules per batch

    Returns
    -------
    List of SMILES strings and batch_no_ls
    """

    batch_df = pd.DataFrame()
    batch_df["ID"] = molids
    batch_df["batch_no"] = [
        molid2BatchNo(
            molid=molid, prefix=prefix, dataset_file=data_fpath, chunksize=chunksize
        )
        for molid in molids
    ]

    batch_no_ls = []
    ids_in_batch = []
    smi_ls = []

    um = batch_df.reset_index().groupby("batch_no", group_keys=True)["ID"].apply(list).items()
    for batch_no, molid_ls in um:
        batch_no_ls.append(batch_no)
        ids_in_batch.append(molid_ls)

    for batch_no, molid_list in zip(batch_no_ls, ids_in_batch):
        try:
            batch_fpath = data_fpath.replace("*", str(batch_no))
        except TypeError as e:
            batch_fpath = data_fpath

        batch_df = pd.read_csv(batch_fpath, index_col="ID")

        print(batch_df)
        for molid in molid_list:
            smi_ls.append(batch_df.loc[molid, "SMILES"])

    return smi_ls


def getChemblMolidSmile(
        chembl_smi_fpath=DATASETS_DIR / 'ChEMBL' /'training_data' / 'dock' / 'ChEMBL_docking_all.csv'
        ):

    chembl_df = pd.read_csv(chembl_smi_fpath)
    molids = chembl_df["ID"].tolist()
    smi_ls = chembl_df["SMILES"].tolist()

    return molids, smi_ls


def cleanChosenMol(
        experiment_ls: list,
        experiment_dir: str,
        save_changes: bool=True,
        logger=None
        ):
    
    """
    Description
    -----------
    Function to clean the chosen mol files incase of restarts, will delete second instance of molecules so be careful
    
    Parameters
    ----------
    experiment_ls (list)        List of experiments to clean the chosen mol_files
    experiment_dir (str)        Directory containing all experiments to clean chosen mol files
    excluded_prefix (str)       Molecules with this prefix will not be selected to the chosen_mol file
                                e.g., not including "CHEMBL" molecules
    
    Returns
    -------
    None
    """        

    if logger is None:
        logger = logging.getLogger(__name__)

    for exp in experiment_ls:
        logger.debug(f"Cleaning {exp} chosen_mol.csv")
        exp_dpath = experiment_dir + exp + '/'
        chosen_mols_fpath = exp_dpath + 'chosen_mol.csv'
        chosen_mol_df = pd.read_csv(chosen_mols_fpath, index_col='ID')

        chosen_mol_df.sort_values(by='Iteration')

        filtered_df = chosen_mol_df.groupby('Iteration', group_keys=False).apply(
            lambda group: group.iloc[10:] if len(group) > 10 else group
                                )

        if save_changes:
            chosen_mol_df.to_csv(
                exp_dpath + 'uncleaned_chosen_mol.csv', index_label='ID')
            filtered_df.to_csv(chosen_mols_fpath, index_label='ID')


def removeTempDirs( 
        experiment_ls: list,
        experiment_dir: str,
        temp_suffix: str='_running',
        logger=None
        ):
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for exp in experiment_ls:
        exp_dpath = experiment_dir + exp + '/'
        path = Path(exp_dpath)
        for item in path.iterdir():
            if item.is_dir() and temp_suffix in item.name:
                logger.info(f"Removing directory: {item}")
                shutil.rmtree(item)


def getDesc4Molid(
        molid_ls: list,
        chembl_fpath = DATASETS_DIR / 'ChEMBL' / 'training_data' / 'desc' / 'rdkit' / 'ChEMBL_rdkit_desc_trimmed.csv',
        pmg_fpath = DATASETS_DIR / 'datasets' / 'PyMolGen' / 'desc' / 'rdkit' / 'PMG_rdkit_desc*',
        logger=None
    ):

    if logger is None:
        logger = logging.getLogger(__name__)

    columns = pd.read_csv(chembl_fpath, index_col='ID').columns
    all_desc_df = pd.DataFrame(columns=columns)


    for n, molid in enumerate(molid_ls):
        if molid.startswith('CHEMBL'):
            desc_df = pd.read_csv(chembl_fpath, index_col='ID')
            try:
                row = desc_df.loc[molid]
            except KeyError as e:
                logger.warning(f"Key warning, couldn't find {e}")
                continue

        elif molid.startswith('PMG-'):
            batch_no = molid2BatchNo(
                molid, 'PMG-', 
                dataset_file=pmg_fpath
                )
            
            desc_fbatch = str(pmg_fpath).replace('*', str(batch_no))
            desc_df = pd.read_csv(desc_fbatch, index_col='ID')
            row = desc_df.loc[molid]
        
        else:
            logger.info('Nothing found here')
        
        all_desc_df.loc[molid] = row

    return all_desc_df


def fig2Img(fig):
    """Convert a Matplotlib figure to a numpy array with RGB channels"""
    import io
    from PIL import Image
    
    # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Convert to PIL Image and then to numpy array
    img = Image.open(buf)
    return np.array(img)


def getTop(df,
            n: int,
            column: str,
            ascending: bool):
    "Function to get the top n number of molecules from a single or list of dataframes"

    if isinstance(df, pd.DataFrame):
        return df.sort_values(by=column, ascending=ascending).head(n)

    elif isinstance(df, list):        
        combined_df = pd.concat(
            [x.sort_values(by=column, ascending=ascending).head(n) for x in df],
            ignore_index=False
        )
        return combined_df.sort_values(by=column, ascending=ascending).head(n)
    
    else:
        raise TypeError("Input must be a DataFrame or a list of DataFrames")


def createGIF(image_ls: list,
               save_path: str,
               save_fname: str='GIF',
               fps: int= 1.0,
               target_size: tuple=(1500, 1500),
               loop: int=0,
               quality:int=95):
    """
    Description
    -----------
    Create GIF from a list of file images
    
    Parameters
    ----------
    image_ls (list)
    save_path (str)
    save_fname (str)
    duration(int)
    
    Returns
    -------
    None
    """
    
    save_path = Path(save_path)
    full_save_path = save_path / f"{save_fname}"

    first_img = Image.open(image_ls[0])
    if target_size is None:
        target_size = first_img.size
    images = []
    for image in image_ls:
        img = Image.open(image).resize(target_size, Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)

        images.append(np.array(img))
        images_array = np.stack(images, axis=0)

    imageio.mimsave(f'{str(full_save_path)}.gif', 
                    images_array, 
                    fps=fps, 
                    loop=loop,
                    quality=quality)

def replaceJSONInfo(root_dir: str, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    root_path = Path(root_dir)

    for file_path in root_path.rglob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            updated = replaceKeyCase(data, "Pearson", "pearson")

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(updated, f, indent=4)

            logger.info(f"Updated: {file_path}")
        except Exception as e:
            logger.warning(f"Skipped (error in file {file_path}): {e}")


def replaceKeyCase(obj, old_key, new_key):

    if isinstance(obj, dict):
        return {
            (new_key if k == old_key else k): replaceKeyCase(v, old_key, new_key)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [replaceKeyCase(item, old_key, new_key) for item in obj]
    else:
        return obj
    
def orderFiles(file_ls: list):
    """
    Orders a list of file paths (strings) numerically based on trailing digits
    in the filename (e.g., batch_1.csv, batch_2.csv, ..., batch_10.csv).
    
    Parameters
    ----------
    file_ls : list
        List of file path strings.

    Returns
    -------
    list
        Sorted list of file path strings.
    """
    def extract_number(filename):
        match = re.findall(r"\d+", Path(filename).stem)
        return int(match[-1]) if match else -1  # fallback to -1 if no number found

    return sorted(file_ls, key=extract_number)

def resolveFiles(pattern: str, allow_single: bool = True, logger=None):
    """
    Resolves file paths from a glob pattern and orders them numerically.
    Supports both multiple matches (wildcards) and a fallback for a single direct path.

    Parameters
    ----------
    pattern : str
        Glob pattern (can include * wildcard).
    allow_single : bool, optional
        If True, fallback to checking if a direct path exists.
    logger : Logger, optional
        Logger object for warnings.

    Returns
    -------
    list
        List of resolved and ordered file paths (as strings).
    """
    files = glob(pattern)
    if not files and allow_single:
        path = Path(pattern)
        if path.exists():
            files = [str(path)]
    if logger and not files:
        logger.warning(f"No files matched pattern: {pattern}")
    return orderFiles(files)

def readConfigJSON(config_fpath: str):
    """Reads config files and returns pathways to relevant data"""

    with open(str(config_fpath), "r") as f:
        data = json.load(f)

    return data