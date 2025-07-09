import pandas as pd
from pathlib import Path
import sys
from rdkit import RDLogger
from glob import glob
from tqdm import tqdm
import re

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

FILE_DIR = str(Path(__file__).parent)
PARENT_DIR = str(Path(__file__).parent.parent)

sys.path.insert(0, PARENT_DIR)
from dataset_functions import Dataset_Formatter

make_training_data = False
make_full_data = True
chunksize = 100000


# MAKING TRAINING DATA SET
if make_training_data:
    print("Generating ChEMBL Data set")
    mk = Dataset_Formatter()

    data = mk._load_data(
        mol_dir="/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/",
        filename="raw_ChEMBL_data.csv",
        pymolgen=False,
        mol_type="smiles",
        chunksize=chunksize,
        retain_ids=True,
        prefix=None,
    )
    print(data)

    print("Generating RDKit descriptors")

    desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set="RDKit")

    print("=============================")
    print(desc_chunks)
    print("=============================")
    print(full_chunks)
    print("=============================")
    print("Filtering Results")

    filt_df = mk._filter_df(chembl=True)

    print("Making final chunks...")
    full, desc = mk._make_final_chunks(
        chunksize=chunksize,
        gen_desc_chunks=True,
        descriptor_set="RDKit",
        save_full_data=False,
        save_desc_data=True,
        full_save_path=None,
        desc_save_path="/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/",
        filename="ChEMBL_rdkit_desc",
    )
    print("=============================")
    print(full)
    print("=============================")
    print(desc)
    print("=============================")
    print("Making docking files...")

    for i, chunks in enumerate(full):
        ch_docking_df = pd.DataFrame()
        ch_docking_df["SMILES"] = chunks["SMILES"]
        ch_docking_df["affinity_dock"] = "N/A"
        ch_docking_df["docking_time"] = "N/A"
    print("=============================")
    print(ch_docking_df)
    print("=============================")

    ch_docking_df.to_csv(
        f"/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/ChEMBL_dock.csv.gz",
        index="ID",
        compression="gzip",
    )

    print("Generating Mordred descriptors...")

    desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set="Mordred")

    print("=============================")
    print(desc_chunks)
    print("=============================")
    print(full_chunks)
    print("=============================")
    print("Filtering Results")

    filt_df = mk._filter_df(chembl=True)

    print("=============================")
    print(desc_chunks)
    print("=============================")
    print(full_chunks)
    print("=============================")
    print("Making final chunks...")

    full, desc = mk._make_final_chunks(
        chunksize=chunksize,
        gen_desc_chunks=True,
        descriptor_set="Mordred",
        save_full_data=False,
        save_desc_data=True,
        full_save_path=None,
        desc_save_path="/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/mordred/",
        filename="ChEMBL_mordred_desc",
    )
    print("=============================")
    print(full)
    print("=============================")
    print(desc)
    print("=============================")

# MAKING EXPERIMENT DATA SET
if make_full_data:
    mol_dir = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/raw/"
    filename = "exp_all.inchi"
    chunksize = 100000
    retain_ids = False
    desc_filepath = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/"
    docking_dir = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/docking/"
    temp_fpath = "/users/yhb18174/Recreating_DMTA/datasets/temp/"
    save_dir = "~/Recreating_DMTA/datasets/temp/full_data/mordred/filtered_results/"

    # premade_chunks = glob(f'/users/yhb18174/Recreating_DMTA/datasets/temp/temp_*.csv.gz')

    mk = Dataset_Formatter()

    # print('Loading full data:\n')

    # data = mk.LoadData(mol_dir=mol_dir,
    #                      filename=filename,
    #                      pymolgen=True,
    #                      prefix='PMG-',
    #                      mol_type='smiles',
    #                      chunksize=chunksize,
    #                      retain_ids=retain_ids,
    #                      temp_files=premade_chunks,
    #                      save_path=temp_fpath,
    #                      remove_temp_files=False)

    # temp_df_ls = glob(f'{desc_filepath}PyMolGen*')

    # desc_fpath_ls, full_fpath_ls = mk.CalcDescriptors(descriptor_set = 'RDKit', csv_list=temp_df_ls, tmp_dir = temp_fpath)

    # print('Filtering DF')

    # full_fpath_ls = glob(f'{temp_fpath}RDKit_full_batch_*.csv.gz')

    # filt_df = mk.FilterMols(full_fpath_ls=full_fpath_ls, save_dir=save_dir, rdkit_or_mordred='RDKit')

    # print('Making Final Chunks')

    # filt_fpath_ls = glob(f'/users/yhb18174/Recreating_DMTA/datasets/temp/full_data/rdkit/filtered_results/RDKit_filtered_results_batch_*.csv.gz')

    # full, desc = mk.MakeFinalChunks(chunksize=chunksize,
    #                                    gen_desc_chunks=True,
    #                                    descriptor_set='RDKit',
    #                                    full_save_path=desc_filepath + 'rdkit/full_data/',
    #                                    desc_save_path=desc_filepath + 'rdkit/',
    #                                    filename='PMG_rdkit',
    #                                    filt_fpath_ls = filt_fpath_ls,
    #                                    index_prefix='PMG')

    file_paths = glob(f"{desc_filepath}rdkit/full_data/PMG_rdkit_*.csv")

    def extract_number(filename):
        """
        Extracts a number from a filename based on a specific pattern.
        """
        match = re.search(r"_(\d+)\.csv", filename)
        return int(match.group(1)) if match else float("inf")

    def sort_files_by_number(file_paths):
        """
        Sorts file paths based on the numerical suffix in the filename.
        """
        return sorted(file_paths, key=extract_number)

    sorted_file_paths = sort_files_by_number(file_paths)
    print(sorted_file_paths)

    for i, chunks in enumerate(
        tqdm(sorted_file_paths, desc="Docking files", unit="chunks")
    ):
        chunks = pd.read_csv(chunks, index_col="ID")
        py_docking_df = pd.DataFrame()
        py_docking_df.index = chunks.index
        py_docking_df["SMILES"] = chunks["SMILES"]
        # py_docking_df['CNN_affinity'] = 'N/A'
        py_docking_df["Affinity(kcal/mol)"] = "N/A"
        py_docking_df.to_csv(f"{docking_dir}PMG_docking_{i+1}.csv", index_label="ID")

    # print('Generating Mordred descriptors')

    # temp_df_ls = glob(f'{desc_filepath}PyMolGen*')

    # desc_chunks, full_chunks = mk.CalcDescriptors(descriptor_set='Mordred', csv_list=temp_df_ls, tmp_dir=temp_fpath)

    # print('Filtering chunks')

    # filt_fpath_ls = glob(f'/users/yhb18174/Recreating_DMTA/datasets/temp/full_data/mordred/filtered_results/Mordred*.csv.gz')
    # full, desc = mk.MakeFinalChunks(chunksize=chunksize,
    #                                    gen_desc_chunks=True,
    #                                    descriptor_set='Mordred',
    #                                    full_save_path=desc_filepath + 'rdkit/full_data/',
    #                                    desc_save_path=desc_filepath + 'rdkit/',
    #                                    filename='PMG_rdkit',
    #                                    filt_fpath_ls = filt_fpath_ls,
    #                                    index_prefix='PMG')
