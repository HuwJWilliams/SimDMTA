import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from mordred import Calculator, descriptors
import openeye as oe
from openeye import oechem
import openbabel as ob
import numpy as np
from multiprocessing import Pool, cpu_count, Process
from pathlib import Path
import tempfile
from io import StringIO
import subprocess
import glob
import time
from tqdm import tqdm
import traceback

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

ROOT_DIR = Path(__file__).parent
PROJ_DIR = str(Path(__file__).parent.parent.parent)


class Dataset_Formatter:
    """
    Description
    -----------
    The class to make and format all of the dataset information for the RecDMTA project.
    """

    def __init__(self):
        return

    def _process_line(self, line):
        """
        Description
        -----------
        Takes inchis from he PyMolGen exp.inchi files and converts them into SMILES strings

        Parameters
        ----------
        line (str)       Line from .inchi file

        Returns
        -------
        SMILES string obtained from inchi
        or
        None if cannot inchi is invalid
        """
        mol = Chem.MolFromInchi(line.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return None

    def _make_chunks(
        self,
        mol_dir: str,
        filename: str,
        retain_ids: bool,
        pymolgen: bool = False,
        prefix: str = "HW-",
        chunksize: int = 100000,
    ):
        """
        Description
        -----------
        Function which takes a file and splits it into chunks. New IDs can be made for the molecules.

        Parameters
        ----------
        mol_dir (str)       Directory which contains the file you wish the split into chunks
        filename (str)      Name of the file which you wish to split into chunks
        retain_ids (bool)   Flag to keep orginal IDs in the file
                            True:  Keep IDs
                            False: Make new IDs
        pymolgen (bool)     Flag to check whether or not the input file originates from PyMolGen
        prefix (str)        Prefix to allocate to the IDs of the molecule, will be followed by a number (e.g., 'HW-1')
        chunksize (int)     Size of chunks to make

        Returns
        -------
        The pd.DataFrame chunks made from the original file
        """
        print("Making Chunks")

        chunks = []

        if pymolgen:
            with open(f"{mol_dir}/{filename}", "r") as file:
                lines = file.readlines()
                print("File open")

            lines_chunks = [
                lines[i : i + chunksize] for i in range(0, len(lines), chunksize)
            ]

            for chunk_idx, chunk_lines in enumerate(lines_chunks):
                chunk_results = []
                id_ls = []

                for line_idx, line in enumerate(chunk_lines):
                    result = self._process_line(line)
                    if result is not None:
                        chunk_results.append(result)
                        id_ls.append(f"{prefix}{len(chunk_results)}")

                chunk_df = pd.DataFrame({"ID": id_ls, "SMILES": chunk_results})
                chunk_df.set_index("ID", inplace=True)
                chunks.append(chunk_df)
                print(f"Chunk {chunk_idx + 1} made ({len(chunk_df)} lines)...")

        else:
            mol_df = pd.read_csv(mol_dir + filename)
            smi_ls = mol_df["SMILES"].tolist()
            id_ls = (
                mol_df["ID"].tolist()
                if retain_ids
                else [f"{prefix}{i+1}" for i in range(len(smi_ls))]
            )

            for i in range(0, len(smi_ls), chunksize):
                chunk_smi_ls = smi_ls[i : i + chunksize]
                chunk_id_ls = id_ls[i : i + chunksize]
                chunk_df = pd.DataFrame({"ID": chunk_id_ls, "SMILES": chunk_smi_ls})
                chunk_df.set_index("ID", inplace=True)
                chunks.append(chunk_df)

        print(f"Made {len(chunks)} chunks")

        return chunks

    def _make_chunks_wrapper(self, args):
        """
        Description
        -----------
        Wrapper function to allow the use of _make_chunks() in multiprocessing tasks

        Parameters
        ----------
        args        Arguments used for the _make_chunks() function:
                    (mol_dir: str,
                     filename: str,
                     retain_ids: bool,
                     pymolgen: bool=False,
                     prefix: str='HW-',
                     chunksize:int=100000)

        Returns
        -------
        Chunks made from the 'filename' provided in args
        """
        return self._make_chunks(*args)

    def _process_mols(
        self,
        mol_type: str,
        filename: str,
        column: str = "SMILES",
        sub_point: int = None,
        core: str = None,
        keep_core: bool = True,
    ):
        """
        Description
        -----------
        Function that takes a list of inchis and canonicalises them

        Parameters
        ----------
        inchi_list (str)        List of inchi strings you wish to convert to their canonical tautomer
        sub_point (int)         Atom number the fragments were added on to (for keeping original core tautomer)
        core (str)              Core you started with and want to maintain
        keep_core (bool)        Flag to set whether or not you want to maintain the previously defined core
                                after canonicalisation

        Returns
        -------
        4 lists:
                canon_mol_list = List of canonical RDKit mol objects
                frag_smi_list  = List of the canonical fragments added onto the original core
                                (only works if keep_core is True, core and sub point are defined)
                canon_smi_list = List of canonical SMILES strings, determined by RDKit
                kekulised_smi_ls = List of canonical kekulised smiles strings as determined by RDKit.
        """

        # Converting all inchis into RDKit mol objects

        mol_converters = {"inchi": Chem.MolFromInchi, "smiles": Chem.MolFromSmiles}

        mol_converter = mol_converters.get(mol_type)

        if mol_converter is None:
            raise ValueError(
                "Unsupported mol_type. Supported types are 'inchi' and 'smiles'."
            )

        mol_list = pd.read_csv(filename, compression="gzip")[column]

        mol_list = [mol_converter(x) for x in mol_list if mol_converter(x) is not None]

        # Canonicalising the SMILES and obtaining the 3 lists produced by the _canonicalise_smiles() function
        results = [
            self._canonicalise_smiles(
                mol, keep_core=keep_core, core_smi=core, sub_point=sub_point
            )
            for mol in mol_list
        ]

        # Isolating each output list
        frag_smi_list = [result[1] for result in results if results[0]]
        canon_smi_list = [result[2] for result in results if results[0]]

        # Converting them to pH 7.4
        ph74_smi_ls = [
            self._adjust_smi_for_ph(smi, phmodel="OpenEye") for smi in canon_smi_list
        ]
        ph74_mol_ls = [Chem.MolFromSmiles(smi) for smi in ph74_smi_ls]
        kekulised_smi_ls = [
            Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in ph74_mol_ls
        ]

        return ph74_mol_ls, frag_smi_list, ph74_smi_ls, kekulised_smi_ls

    def _process_mols_wrapper(self, args):
        """
        Description
        -----------
        Wrapper function to allow the use of _process_mols() in multiprocessing tasks

        Parameters
        ----------
        args        Arguments used for the _make_chunks() function:
                    (mol_type: str,
                     filename: str,
                     column: str='SMILES',
                     sub_point: int=None,
                     core: str=None,
                     keep_core: bool=True)

        Returns
        -------
        Processed mols made from the 'filename' provided in args
        """
        return self._process_mols(*args)

    def _canonicalise_smiles(
        self, mol: object, keep_core=True, core_smi: str = None, sub_point: int = None
    ):
        """
        Description
        -----------
        Canonicalising function which deals with tautomeric forms different from that of the original core.
        Here the idea is to canonicalise with RDKit and check the molecule against the original core.
        If original core is maintained, the function returns the canonical SMILES string.
        If not, the fragment is isolated, canonicalised and stiched back onto the original core

        Parameters
        ----------
        smiles (str)            SMILES string of molecule to be checked
        keep_core (bool)        Flag to keep the original specied core, or allow it
                                to be tautomerised
        core (str)              SMILES string of the core you want to check against
        sub_point (int)         Point to stitch the fragment onto the sub_core
        sub_core (str)          SMILES string of the core you want to add the fragments onto
                                (leave empty is this is just the previous core)

        Returns
        -------
        has_core (bool)             Tells you whether or not the molecule has one of the tautomeric forms of
                                    the specified. True means that one of the forms is present
        frag_smi (str)              SMILES string of the fragment added onto the core
        canon_smi (str)             SMILES string of the canonical form of the molecule after added onto
                                    sub_core
        canon_mol (rdkit.mol)       RDKit molecule object mde fromm the canonical SMILES string
        """

        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)
        canon_smi = Chem.MolToSmiles(canon_mol)

        # If maintaining the original core is not an issue then just canonicalise as usual,
        # Does not show fragment SMILES strings
        if not keep_core:
            return True, None, canon_smi, canon_mol

        if core_smi is None:
            raise ValueError("Invalid core SMILES string provided.")

        core_mol = Chem.MolFromSmiles(core_smi)

        # Initialising the enumerator, canonicalising the molecule and obtaining all tautomeric
        # forms of the original core
        core_tauts = enumerator.Enumerate(core_mol)

        # Checking if the mol has the original core
        if canon_mol.HasSubstructMatch(core_mol):
            # If so, return the canonical smiles, fragment and core flag
            frag_mol = Chem.ReplaceCore(mol, core_mol)
            frag_smile = Chem.MolToSmiles(frag_mol)

            return True, frag_smile, canon_smi, canon_mol

        # If it doesnt have the original core, check the tautomeric forms
        for taut in core_tauts:

            # If it has one of the tautometic forms, substitute the core with
            # a dummy atom
            if canon_mol.HasSubstructMatch(taut):
                frag_mol = Chem.ReplaceCore(mol, taut)
                dummy_idx = next(
                    atom.GetIdx()
                    for atom in frag_mol.GetAtoms() is atom.GeySymbol() == "*"
                )
                neighbour_idx = next(
                    bond.GetOtherAtomIdx(dummy_idx)
                    for bond in frag_mol.GetAtomWithIdx(dummy_idx).GetBonds()
                )

                frag_mol = Chem.EditableMol(frag_mol)
                frag_mol.RemoveAtom(dummy_idx)
                frag_mol = frag_mol.GetMol()
                frag_smi = Chem.MolToSmiles(frag_mol)

                combined_mol = Chem.CombineMols(frag_mol, core_smi)
                combined_mol = Chem.EditableMole(combined_mol)
                sub_atom = sub_point + frag_mol.GetNumAtoms() - 1

                combined_mol.AddBond * neighbour_idx, sub_atom, Chem.rdchem.BondType.SINGLE
                final_mol = Chem.RemoveHs(combined_mol.GetMol())
                final_smi = Chem.MolToSmiles(final_mol)

                return True, frag_smi, final_smi, final_mol

        return False, None, canon_smi, canon_mol

    def _kekulise_smiles(self, mol):
        """
        Description
        -----------
        Function to Kekulise SMILES

        Parameters
        ----------
        mol (rdkit.mol)     RDKit mol object

        Returns
        -------
        Kekulised SMILES string of molecule
        """

        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def _conv_df_to_str(self, df: pd.DataFrame, **kwargs):
        """
        Description
        -----------
        Converts a Data Frame to a string

        Parameters
        ----------
        **kwargs    Arguments for the pd.DataFrame.to_csv() function (e.g., index, compression, etc.)

        Returns
        -------
        String representation of a dataframe
        """

        string = StringIO()
        df.to_csv(string, **kwargs)
        return string.getvalue()

    def _apply_lilly_rules(
        self,
        df: pd.DataFrame = None,
        smiles: list = [],
        cleanup: bool = True,
        run_in_temp_dir: bool = True,
        smi_input_filename: str = None,
        lilly_rules_script: str = str(ROOT_DIR)
        + "/Lilly-Medchem-Rules/Lilly_Medchem_Rules.rb",
    ):
        """
        Description
        -----------
        Apply Lilly rules to SMILES in a list or a DataFrame.

        Parameters
        ----------
        df (pd.DataFrame)           DataFrame containing SMILES
        smiles (list)               List of SMILES to aply the Lilly-MedChem-Rules to
        cleanup (bool)              Flag to clean up any temporary files and directories made by the function
        run_in_temp_dir (bool)      Flag to run the script in a temporary directory (recommended), or the
                                    current working directory
        lilly_rules_script (str)    Pathway to "Lilly_Medchem_Rules.rb" script (found in the Lill-MedChem-Rules submodule)

        Returns
        -------
        pd.DataFrame
            DataFrame containing results of applying Lilly's rules to SMILES, including pass/fail and warnings

        Example
        -------
        >>> apply_lilly_rules(smiles=['CCCCCCC(=O)O', 'CCC', 'CCCCC(=O)OCC', 'c1ccccc1CC(=O)C'])
                    SMILES       SMILES_Kekule  Lilly_rules_pass      Lilly_rules_warning  Lilly_rules_SMILES
        0     CCCCCCC(=O)O        CCCCCCC(=O)O              True        D(80) C6:no_rings        CCCCCCC(=O)O
        1              CCC                 CCC             False     TP1 not_enough_atoms                 CCC
        2     CCCCC(=O)OCC        CCCCC(=O)OCC              True  D(75) ester:no_rings:C4        CCCCC(=O)OCC
        3  c1ccccc1CC(=O)C  CC(=O)CC1=CC=CC=C1              True                     None  CC(=O)CC1=CC=CC=C1
        """

        lilly_rules_script_path = Path(lilly_rules_script)

        if not lilly_rules_script_path.is_file():
            raise FileNotFoundError(
                f"Cannot find Lilly rules script (Lilly_Medchem_Rules.rb) at: {lilly_rules_script_path}"
            )

        smi_file_txt = self._conv_df_to_str(
            df[["Kekulised_SMILES", "ID"]], sep=" ", header=False, index=False
        )
        # Optionally set up temporary directory:
        if run_in_temp_dir:
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = temp_dir.name + "/"
        else:
            run_dir = "./"

        # If filename given, save SMILES to this file:
        if smi_input_filename is not None:
            with open(run_dir + smi_input_filename, "w") as temp:
                temp.write(smi_file_txt)

        # If no filename given just use a temporary file:
        else:
            # Lilly rules script reads the file suffix so needs to be .smi:
            temp = tempfile.NamedTemporaryFile(mode="w+", suffix=".smi", dir=run_dir)
            temp.write(smi_file_txt)
            # Go to start of file:
            temp.seek(0)

        # Run Lilly rules script
        lilly_results = subprocess.run(
            [f"cd {run_dir}; ruby {lilly_rules_script} {temp.name}"],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if lilly_results.stderr.decode("utf-8") != "":
            print("WARNING: {}".format(lilly_results.stderr.decode("utf-8")))
        lilly_results = lilly_results.stdout.decode("utf-8")

        # Process results:
        passes = []
        if lilly_results != "":
            for line in lilly_results.strip().split("\n"):

                # Record warning if given:
                if " : " in line:
                    smiles_molid, warning = line.split(" : ")
                else:
                    smiles_molid = line.strip()
                    warning = None
                smiles, molid = smiles_molid.split(" ")
                passes.append([molid, warning, smiles])

        # Get reasons for failures:
        failures = []

        # Maybe change 'r' to 'w+'
        for bad_filename in glob.glob(run_dir + "bad*.smi"):
            with open(bad_filename, "r") as bad_file:
                for line in bad_file.readlines():
                    line = line.split(" ")
                    smiles = line[0]
                    molid = line[1]
                    warning = " ".join(line[2:]).strip(": \n")
                    failures.append([molid, warning, smiles])

        # Close and remove tempfile:
        # (Do this even if run in a temporary directory to prevent warning when
        # script finishes and tries to remove temporary file at that point)
        if smi_input_filename is None:
            temp.close()

        if run_in_temp_dir:
            temp_dir.cleanup()
        elif cleanup:
            subprocess.run(["rm -f ok{0,1,2,3}.log bad{0,1,2,3}.smi"], shell=True)

        # Convert to DataFrame:
        df_passes = pd.DataFrame(
            passes, columns=["ID", "Lilly_rules_warning", "Lilly_rules_SMILES"]
        )
        #                .set_index('ID', verify_integrity=True)
        df_passes.insert(0, "Lilly_rules_pass", True)

        df_failures = pd.DataFrame(
            failures, columns=["ID", "Lilly_rules_warning", "Lilly_rules_SMILES"]
        )
        #                .set_index('ID', verify_integrity=True)

        df_failures.insert(0, "Lilly_rules_pass", False)

        df_all = pd.concat([df_passes, df_failures], axis=0)

        df_out = pd.merge(df, df_all, on="ID", how="inner")

        print(df[~df["ID"].isin(df_out["ID"])])

        # Check all molecules accounted for:
        if len(df_out) != len(df):
            raise ValueError(
                "Some compounds missing, {} molecules input, but {} compounds output.".format(
                    len(df), len(df_out)
                )
            )

        return df_out.set_index("ID")

    def _adjust_smi_for_ph(self, smi: str, ph: float = 7.4, phmodel: str = "OpenEye"):
        """
        Description
        -----------
        Function to adjust smiles strings for a defined pH value
        canon_mol_list = [result[3] for result in results if results[0]]
        kekulised_smi_ls = [self._kekulise_smiles(mol) for mol in canon_mol_list]
        Parameters
        ----------
        smi (str)       SMILES string you want to adjust
        ph (float)      pH you want to adjust SMILES to
        phmodel (str)   pH model used, use either OpenEye or OpenBabel

        Returns
        -------
        SMILES string of adjusted molecule
        """

        # OpenEye pH model needs a pH of 7.4
        if phmodel == "OpenEye" and ph != 7.4:
            raise ValueError("Cannot use OpenEye pH conversion for pH != 7.4")

        # Use OpenBabel for pH conversion
        # NEED TO PIP INSTALL OBABEL ON phd_env
        if phmodel == "OpenBabel":
            ob_conv = ob.OBConversion()
            ob_mol = ob.OBMol()
            ob_conv.SetInAndOutFormats("smi", "smi")
            ob_conv.ReadString(ob_mol, smi)
            ob_mol.AddHydrogens(
                False, True, ph  # <- only add polar H  # <- correct for pH
            )
            ph_smi = ob_conv.WriteString(ob_mol, True)  # <- Trim White Space

            # Check that pH adjusted SMILES can be read by RDKit,
            # if not return original SMILES
            if Chem.MolFromSmiles(ph_smi) is None:
                ph_smi = smi

        # Use OpenEye for pH conversion
        elif phmodel == "OpenEye":
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, smi)
            oe.OESetNeutralpHModel(mol)
            ph_smi = oechem.OEMolToSmiles(mol)

        return ph_smi

    def _calculate_oe_LogP(self, smi: str):
        """
        Description
        -----------
        Function to calculate the LogP using OpenEye

        Parameters
        ----------
        smi (str)       SMILES string you wish to calculate LogP for

        Returns
        -------
        The oe_logp of input smile
        """

        # Initialise converter
        mol = oe.OEGraphMol()

        # Check is smile gives valid molecule object
        if not oe.OESmilesToMol(mol, smi):
            print("ERROR: {}".format(smi))
        else:
            # Calculate logP
            try:
                logp = oe.OEGetXLogP(mol, atomxlogps=None)
            except RuntimeError:
                print(smi)
        return logp

    def _get_descriptors(self, mol, missingVal=None, descriptor_set: str = "RDKit"):
        """
        Description
        -----------
        Function to get the descriptors for a molecule. Can get RDKit or Mordred descriptors

        Parameters
        ----------
        mol (object)            RDKit molecule object you wish to calculate descriptors for
        missingVal (int)        If descriptor value cannot be calculated, enter missingVal instead (keep as None)
        descriptor_set (str)    Option to obtain RDKit or Mordred descriptors

        Returns
        -------
        Dictionary containing all of the descriptors and their values:

        {
            desc_1: val1,
            desc_2: val2,
            ...
        }

        """

        res = {}
        if descriptor_set == "RDKit":
            for name, func in Descriptors._descList:
                # some of the descriptor fucntions can throw errors if they fail, catch those here:
                try:
                    val = func(mol)
                except:
                    # print the error message:
                    import traceback

                    traceback.print_exc()
                    # and set the descriptor value to whatever missingVal is
                    val = missingVal
                res[name] = val

        elif descriptor_set == "Mordred":
            calc = Calculator(descriptors, ignore_3D=True)

            try:
                desc = calc(mol)

                for name, value in desc.items():
                    res[str(name)] = float(value) if value is not None else missingVal

            except Exception as e:
                traceback.print_exc()
                for descriptor in calc.descriptors:
                    res[str(descriptor)] = missingVal

        return res

    def _get_descriptors_wrapper(self, args):
        """
        Description
        -----------
        Wrapper function to allow the use of _get_descriptors() in multiprocessing tasks

        Parameters
        ----------
        args        Arguments used for the _make_chunks() function:
                    (mol,
                     missingVal=None,
                     descriptor_set: str='RDKit')

        Returns
        -------
        Dictionary of descriptors for a given molecule
        """
        return self._get_descriptors(*args)

    def _gen_descriptors(self, mol, descriptor_set: str, missingVal=None):
        if descriptor_set == "RDKit":
            for name, func in Descriptors._descList:
                try:
                    val = func(mol)
                except Exception:
                    traceback.print_exc()
                    val = missingVal
                yield name, val

        elif descriptor_set == "Mordred":
            calc = Calculator(descriptors, ignore_3D=True)

            try:
                desc = calc(mol)
                for name, value in desc.items():
                    yield str(name), (
                        np.float32(value) if value is not None else missingVal
                    )
            except Exception as e:
                traceback.print_exc()
                for descriptor in calc.descriptors:
                    yield str(descriptor), missingVal

    def _descriptor_worker(self, smi, descriptor_set, missingVal=None):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            descriptor_dict = dict(
                self._gen_descriptors(mol, descriptor_set, missingVal)
            )
            return descriptor_dict
        return {}

    def LoadData(
        self,
        mol_dir: str,
        filename: str,
        prefix: str,
        pymolgen: bool,
        mol_type: str,
        column: str = "SMILES",
        chunksize: int = 10000,
        retain_ids: bool = False,
        core: str = None,
        sub_point: int = None,
        keep_core: bool = False,
        save_chunks: bool = True,
        save_name: str = "PyMolGen",
        save_path: str = f"{PROJ_DIR}/datasets/temp",
        temp_files: list = None,
        remove_temp_files: bool = True,
    ):
        """
        Description
        -----------
        Function to load all of the data from provided files, make chunks, process mols and save them to a .csv.gz file in mol_dir

        Parameters
        ----------
        mol_dir (str)               Directory which the molecule file is kept
        filename (str)              Name of molecule file
        prefix (str)                Prefix for setting new IDs for molecules
        pymolgen (bool)             Flag to check whether or not the data originates from PyMolGen
        mol_type (str)              Molecules presented as 'smiles' or 'inchi' strings
        column (str)                Column in the dataframe to process (keep as 'SMILES')
        chunksize (int)             Size of chunks to keep the data in
        retain_ids (bool)           Flag to retain original index/ids from the data or make new ID prefixes
        core (str)                  SMILES string core to force keep on the molecules (works for only monosubstituted core)
        sub_point (int)             Atom number for which substitutions have been made on the 'core' argument
        keep_core (bool)            Flag to allow the forcing of 'core' on molecules
        save_chunks (bool)          Flag to save chunks
        save_name (str)             Prefix to name the saved chunk_files under
        save_path (str)             Path to save temporary files to
        temp_files (list)           Option to use presaved temp files. Provide a list of filepaths

        Returns
        -------
        List of final file pathways to .csv.gz files
        """
        print("\n=======================\nLoading Data\n=======================\n")

        self.final_file_ls = []

        if temp_files is None:
            temp_files = []

            chunks = self._make_chunks(
                mol_dir, filename, retain_ids, pymolgen, prefix, chunksize
            )

            # Save chunks to temp .csv.gz files
            print("Saving chunks:")
            for i, chunk in enumerate(chunks):
                print(i + 1)
                tmp_file = f"{save_path}/temp_chunk_{i+1}.csv.gz"
                chunk.to_csv(tmp_file, index="ID", compression="gzip")
                temp_files.append(tmp_file)

        else:
            chunks = [
                pd.read_csv(tmp_file, index_col="ID", compression="gzip")
                for tmp_file in temp_files
            ]

        arguments = [
            (mol_type, file, column, sub_point, core, keep_core) for file in temp_files
        ]

        for i, (chunk, args) in enumerate(
            tqdm(
                zip(chunks, arguments),
                desc="Processing chunks",
                unit="chunks",
                total=len(chunks),
            )
        ):
            processed_mols = self._process_mols_wrapper(args)
            canon_mol_ls, frag_smi_ls, canon_smi_ls, kek_smi_ls = processed_mols

            data = {
                "ID": chunk.index,
                "Mol": canon_mol_ls,
                "Frag_SMILES": frag_smi_ls,
                "SMILES": canon_smi_ls,
                "Kekulised_SMILES": kek_smi_ls,
            }

            smi_df = pd.DataFrame(data)
            lilly_smi_df = self._apply_lilly_rules(smi_df)

            if remove_temp_files:
                for file in temp_files:
                    file = Path(file)
                    if file.exists():
                        file.unlink()

            if save_chunks:
                save_to = f"{save_path}{save_name}_{i+1}.csv.gz"
                self.final_file_ls.append(save_to)
                lilly_smi_df.drop(columns=["Mol"], inplace=True)
                lilly_smi_df.to_csv(save_to, index="ID", compression="gzip")
                print(f"Made chunk with path:\n{save_to}")

        return self.final_file_ls

    def _calc_desc_mp(self, chunk: pd.DataFrame, descriptor_set: str):
        rows = chunk.to_dict("records")

        descriptors = [
            (
                Chem.MolFromSmiles(row["SMILES"]),
                None,
                descriptor_set,
            )
            for row in rows
        ]

        with Pool() as pool:
            desc_mp_item = pool.map(self._get_descriptors_wrapper, descriptors)

        return desc_mp_item

    def _gen_desc_mp(self, chunk: pd.DataFrame, descriptor_set: str, missingVal=None):
        rows = chunk.to_dict("records")

        args = [(row["SMILES"], descriptor_set, missingVal) for row in rows]

        with Pool() as pool:
            desc_mp_item = pool.starmap(self._descriptor_worker, args)

        return desc_mp_item

    def CalcDescriptors(
        self, descriptor_set: str, csv_list: list = None, tmp_dir: str = None
    ):
        """
        Description
        -----------
        Function to calculate descriptors for a whole dataset

        Parameters
        ----------
        descriptor_set (str)    Choose the descriptor set you want to generate in the (_get_descriptors() function)
                                either 'RDKit' or 'Mordred'
        csv_list (list)         List of pregenerated csv files coming from the LoadData function.
        tmp_dir (str)           Save results in the tmp_dir

        Returns
        -------
        1: List of file pathways to the descriptor files
        2: List of file pathways to the full files
        """
        print(
            "\n=======================\nCalculating descriptors\n=======================\n"
        )
        # Setting up temporary df so to not save over self.smi_df
        if csv_list is not None:
            self.final_file_ls = csv_list

        self.desc_fpath_ls = []
        self.full_fpath_ls = []

        for i, file in enumerate(
            tqdm(self.final_file_ls, desc="Processing chunks", unit="chunks")
        ):
            tmp_df = pd.read_csv(file, index_col="ID", compression="gzip")
            chunks = np.array_split(tmp_df, np.ceil(len(tmp_df) / 1000))
            # Getting the descriptors for each mol object and saving the dictionary
            # in a column named descriptors

            desc_ls = []
            for chunk in chunks:
                desc_mp_item = self._gen_desc_mp(chunk, descriptor_set=descriptor_set)
                desc_ls.extend(desc_mp_item)

            # Making a new pd.Dataframe with each descriptor as a column, setting the
            # index to match self.smi_df (or tmp_df)
            tmp_df["Descriptors"] = desc_ls

            if descriptor_set == "RDKit":
                desc_df = pd.DataFrame(
                    tmp_df["Descriptors"].tolist(),
                    columns=[d[0] for d in Descriptors.descList],
                )

            elif descriptor_set == "Mordred":
                calc = Calculator(descriptors, ignore_3D=True)
                desc_df = pd.DataFrame(
                    tmp_df["Descriptors"].tolist(),
                    columns=[str(d) for d in calc.descriptors],
                )

            desc_df["ID"] = tmp_df.index.tolist()
            desc_df = desc_df.set_index("ID")

            if "nARing" in desc_df.columns:
                desc_df.rename(columns={"naRing": "NumAromaticRings"}, inplace=True)
            if "MW" in desc_df.columns:
                desc_df.rename(columns={"MW": "MolWt"}, inplace=True)
            if "NumHAcceptors" not in desc_df.columns:
                desc_df = desc_df.rename(columns={"nHBAcc": "NumHAcceptors"})
            if "NumHDonors" not in desc_df.columns:
                desc_df = desc_df.rename(columns={"nHBDon": "NumHDonors"})

            desc_filename = f"{tmp_dir}{descriptor_set}_desc_batch_{i+1}.csv.gz"
            desc_df.to_csv(desc_filename, index="ID", compression="gzip")
            self.desc_fpath_ls.append(desc_filename)

            # Concatenating the two dfs to give the full set of descriptors and SMILES
            batch_df = pd.concat([tmp_df, desc_df], axis=1, join="inner").drop(
                columns=["Descriptors"]
            )

            batch_df["oe_logp"] = batch_df["SMILES"].apply(self._calculate_oe_LogP)
            batch_df["PFI"] = batch_df["NumAromaticRings"] + batch_df["oe_logp"]

            full_filename = f"{tmp_dir}{descriptor_set}_full_batch_{i+1}.csv.gz"
            batch_df.to_csv(full_filename, index="ID", compression="gzip")
            self.full_fpath_ls.append(full_filename)

        return self.desc_fpath_ls, self.full_fpath_ls

    def FilterMols(
        self,
        rdkit_or_mordred: str,
        mw_budget: int = 600,
        n_arom_rings_limit: int = 3,
        PFI_limit: int = 8,
        remove_3_membered_rings: bool = True,
        remove_4_membered_rings: bool = True,
        num_h_acc: int = 11,
        num_h_don: int = 6,
        pass_lilly_rules: bool = True,
        chembl: bool = False,
        save_dir: str = None,
        chunksize: int = 1000,
        full_fpath_ls: list = [],
    ):
        """
        Description
        -----------
        Function to filter undesirable molecules from the dataset

        Parameters
        ----------
        rdkit_or_mordred (str)          String to save file names
        mw_budget (int)                 Setting a molecular weight budget for molecules
        n_arom_rings_limit (int)        Setting a limit for the number of aromatic rings for molecule
        PFI_limit (int)                 Setting a PFI limit for molecules (need to implement after
                                        OpenEye License comes)
        remove_*_membered_rings (bool)  Flag to remove 3 or 4 membered cycles
        pass_lilly_rules (bool)         Flag to check if molecules pass the LillyMedChemRules

        Returns
        -------
        A list of file pathways to filtered chunks
        """

        print(
            "\n=======================\nFiltering Molecules\n=======================\n"
        )

        self.filt_fpath_ls = []
        if full_fpath_ls:
            self.full_fpath_ls = full_fpath_ls
        else:
            print("No files to process")
            return []

        for i, file in enumerate(
            tqdm(self.full_fpath_ls, desc="Filtering chunks", unit="chunks")
        ):
            chunk_reader = pd.read_csv(
                file, index_col="ID", compression="gzip", chunksize=chunksize
            )
            filtered_chunks = []

            for df in chunk_reader:
                if not chembl:
                    # Obtaining all molecules which pass the defined filters
                    all_passing_mols = df[
                        (df["MolWt"] <= mw_budget)
                        & (df["NumAromaticRings"] <= n_arom_rings_limit)
                        & (df["PFI"] <= PFI_limit)
                        & (df["Lilly_rules_pass"] == pass_lilly_rules)
                        & (df["NumHDonors"] <= num_h_don)
                        & (df["NumHAcceptors"] <= num_h_acc)
                    ]

                    all_passing_mols = all_passing_mols.copy()

                    all_passing_mols.loc[:, "Mol"] = all_passing_mols[
                        "Lilly_rules_SMILES"
                    ].apply(lambda x: Chem.MolFromSmiles(x))
                    all_passing_mols = all_passing_mols[all_passing_mols["Mol"].notna()]

                    filtered_smi = []
                    for index, rows in all_passing_mols.iterrows():
                        for mol in rows["Mol"].GetRingInfo().AtomRings():
                            if (remove_3_membered_rings and len(mol) == 3) or (
                                remove_4_membered_rings and len(mol) == 4
                            ):
                                filtered_smi.append(rows["SMILES"])

                    filtered_smi_set = set(filtered_smi)
                    filtered_results = all_passing_mols[
                        ~all_passing_mols["SMILES"].isin(filtered_smi_set)
                    ]

                else:
                    filtered_results = df

                filtered_results = filtered_results.drop(columns=["Mol"])
                filtered_chunks.append(filtered_results)

                full_filtered_df = pd.concat(filtered_chunks)

                filt_results_fpath = (
                    f"{save_dir}{rdkit_or_mordred}_filtered_results_batch_{i+1}.csv.gz"
                )
                full_filtered_df.to_csv(
                    filt_results_fpath, index="ID", compression="gzip"
                )
                self.filt_fpath_ls.append(filt_results_fpath)

        return self.filt_fpath_ls

    def MakeFinalChunks(
        self,
        chunksize: int,
        gen_desc_chunks: bool = False,
        descriptor_set: str = "RDKit",
        full_save_path: str = None,
        desc_save_path: str = None,
        filename: str = None,
        index_prefix: str = "HW",
        filt_fpath_ls: list = [],
    ):
        """
        Description
        -----------
        Function to make workable pd.DataFrame chunks of data

        Parameters
        ----------
        df (pd.DataFrame)       Data Frame you which to split into chunks
        chunksize (int)         Number of rows you want in each chunk of your Data Frame
        save_data (bool)        Flag to save the chunks
        save_path (str)         Path to save the chunks to
        filename (str)          Name to save chunks as, function will number them for you
        index_prefix (str)      Prefix to set new index with
        filt_fpath_ls (list)    List of pre-filtered csv fpaths to make into final chunks

        Returns
        -------
        1: List of pathways to the full filtered files
        2: List of pathways to the filtered descriptor files
        """
        print(
            "\n=======================\nMaking Final Chunks\n=======================\n"
        )

        if filt_fpath_ls:
            self.filt_fpath_ls = filt_fpath_ls

        not_full_df = pd.DataFrame()
        full_fpath_ls = []
        desc_fpath_ls = []
        global_index = 0
        chunk_counter = 1

        for file_index, file in enumerate(
            tqdm(self.filt_fpath_ls, desc="Making final chunks", unit="chunks")
        ):
            df = pd.read_csv(file, index_col="ID", compression="gzip")
            concat = pd.concat([df, not_full_df], ignore_index=True)

            while len(concat) > chunksize:
                full_chunk = concat.iloc[:chunksize]
                full_chunk.index = [
                    f"{index_prefix}-{i+1}"
                    for i in range(global_index, global_index + chunksize)
                ]
                global_index += chunksize
                fpath = f"{full_save_path}{filename}_{chunk_counter}.csv.gz"
                full_chunk.to_csv(fpath, compression="gzip", index_label="ID")
                print(f"\nWritten {len(full_chunk)} molecules to {fpath}")

                full_fpath_ls.append(fpath)
                chunk_counter += 1
                concat = concat.iloc[chunksize:].reset_index(drop=True)

            not_full_df = concat

        if not_full_df.shape[0] > 0:
            not_full_df.index = [
                f"{index_prefix}-{i+1}"
                for i in range(global_index, global_index + len(not_full_df))
            ]
            fpath = f"{full_save_path}{filename}_{chunk_counter}.csv.gz"
            not_full_df.to_csv(fpath, index_label="ID", compression="gzip")
            print(f"Written {len(not_full_df)} molecules to {fpath}")
            full_fpath_ls.append(fpath)
            chunk_counter += 1

            print(
                f"\nWritten {len(full_fpath_ls)} files containing full data.\nFilepath:{full_save_path}\nChunksize: {chunksize}.\n"
            )

        if gen_desc_chunks:
            if descriptor_set == "RDKit":
                columns, _ = zip(*Descriptors.descList)
            if descriptor_set == "Mordred":
                calc = Calculator(descriptors, ignore_3D=True)
                desc_names = [str(desc) for desc in calc.descriptors]
                columns = []
                for desc in desc_names:
                    if desc == "naRing":
                        columns.append("NumAromaticRings")
                    elif desc == "MW":
                        columns.append("MolWt")
                    elif desc == "nHBAcc":
                        columns.append("NumHAcceptors")
                    elif desc == "nHBDon":
                        columns.append("NumHDonors")
                    else:
                        columns.append(desc)

            for i, file in enumerate(full_fpath_ls):
                df = pd.read_csv(file, index_col="ID", compression="gzip")
                full_desc = df.loc[:, columns]
                fpath = f"{desc_save_path}{filename}_desc_{i+1}.csv.gz"
                full_desc.to_csv(fpath, index_label="ID", compression="gzip")
                desc_fpath_ls.append(fpath)
            print(
                f"Written {len(desc_fpath_ls)} files containing descriptor data.\nFilepath:{desc_save_path}\nChunksize: {chunksize}."
            )

        return full_fpath_ls, desc_fpath_ls


class Dataset_Accessor:
    def __init__(
        self,
        original_path: str,
        temp_suffix: str = ".tmp",
        wait_time: int = 30,
        max_wait: int = 21600,
    ):

        self.original_path = Path(original_path)
        self.temp_path = self.original_path.with_suffix(
            temp_suffix + self.original_path.suffix
        )
        self.wait_time = wait_time
        self.max_wait = max_wait

    def get_exclusive_access(
        self,
        original_path: str = None,
        temp_path: str = None,
        wait_time: int = None,
        max_wait: int = 21600,
    ):
        """
        Description
        -----------
        Function to gain exclusing access to a file

        Parameters
        ----------
        original_path (str)     File name and its pathway
        temp_path (str)         Name of the temporary path you want to name the file as for exclusive access
        wait_time (int)         Time in between attempts for access to file in seconds
        max_wait (int)          Maximum time the file will wait to gain access in seconds (default is 6 hours)
        """

        if original_path is None:
            original_path = self.original_path
        if temp_path is None:
            temp_path = self.temp_path
        if wait_time is None:
            wait_time = self.wait_time

        waited = 0
        while True:
            try:
                original_path.rename(temp_path)
                return str(temp_path)

            except FileNotFoundError:
                print(
                    f'File "{original_path.stem} is in use.\nRetrying in {wait_time} seconds...'
                )

            except Exception as e:
                print(
                    f"An error occurred while renaming {original_path.stem} for exclusive access:\n{e}"
                )
                return None

            time.sleep(wait_time)
            waited += wait_time
            if waited > max_wait:
                print(f"Reached maximum waiting time on file:\n{original_path.stem}")
                return None

    def release_file(self, original_path: str = None, temp_path: str = None):

        if original_path is None:
            original_path = self.original_path
        if temp_path is None:
            temp_path = self.temp_path

        try:
            temp_path.rename(original_path)
            print(f"File {temp_path.stem} renamed back to {original_path.stem}")
        except Exception as e:
            print(
                f"An error occurred while renaming {temp_path.stem} back to {original_path.stem}:\n{e}"
            )

    def edit_df(
        self,
        column_to_edit: str,
        df: pd.DataFrame = None,
        df_path: str = None,
        index_col: str = "ID",
        idxs_to_edit: list = None,
        vals_to_enter: list = None,
        data_dict: dict = None,
    ):

        if df is not None:
            self.df = df

        if df_path is None:
            df_path = self.temp_path

        if df is None:
            self.df = pd.read_csv(df_path, index_col=index_col)

        if idxs_to_edit is not None and vals_to_enter is not None:
            for idx, val in zip(idxs_to_edit, vals_to_enter):
                self.df.loc[idx, column_to_edit] = str(val)

        elif data_dict is not None:
            for idx, val in data_dict.items():
                self.df.loc[idx, column_to_edit] = str(val)

        else:
            raise ValueError(
                "Either data_dict, idx_to_edit, or vals_to_enter were not entered"
            )

        self.df.to_csv(df_path, index_label=index_col)
