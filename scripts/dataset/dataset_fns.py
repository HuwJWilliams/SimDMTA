import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import openeye as oe
from openeye import oechem
import openbabel as ob
from multiprocessing import Pool, cpu_count
from pathlib import Path
import tempfile
from io import StringIO
import subprocess
import glob
import time
import logging
from filelock import FileLock

ROOT_DIR = Path(__file__).parent

class DatasetFormatter:
    def __init__(self, run_dir: str = str(ROOT_DIR), logger=None, log_level="DEBUG"):

        self.run_dir = run_dir
        self.lilly_smi_df = None

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

            self.logger.info(f"Initialised logger: {log_path}")


    def _process_inchi(
        self,
        mol_type: str,
        mol_list: str,
        sub_point: int = None,
        core: str = None,
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
        3 lists:
                canon_mol_list = List of canonical RDKit mol objects
                frag_smi_list  = List of the canonical fragments added onto the original core
                                 (only works if keep_core is True, core and sub point are defined)
                canon_smi_list = List of canonical SMILES strings, determined by RDKit
        """

        # Converting all inchis into RDKit mol objects

        if mol_type == "inchi":
            mol_list = [
                Chem.MolFromInchi(x)
                for x in mol_list
                if Chem.MolFromInchi(x) is not None
            ]
        elif mol_type == "smiles":
            mol_list = [
                Chem.MolFromSmiles(x)
                for x in mol_list
                if Chem.MolFromSmiles(x) is not None
            ]

        # Canonicalising the SMILES and obtaining the 3 lists produced by the _canonicaliseSmiles() function
        results = [
            self._canonicaliseSmiles(
                mol, keep_core=True, core_smi=core, sub_point=sub_point
            )
            for mol in mol_list
        ]

        # Isolating each output list
        frag_smi_list = [result[1] for result in results if results[0]]
        canon_smi_list = [result[2] for result in results if results[0]]
        canon_mol_list = [Chem.MolFromSmiles(x) for x in canon_smi_list]
        kekulised_smi_ls = [self._kekuliseSmiles(mol) for mol in canon_mol_list]

        return canon_mol_list, frag_smi_list, canon_smi_list, kekulised_smi_ls

    def _canonicaliseSmiles(
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
        canon_smi (str)             SMILES string of the canonical form of the molecule after added onto
                                    sub_core
        frag_smi (str)              SMILES string of the fragment added onto the core
        """

        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)

        # If maintaining the original core is not an issue then just canonicalise as usual,
        # Does not show fragment SMILES strings
        if not keep_core:
            has_core = True
            frag_smi = None
            canon_smi = Chem.MolToSmiles(canon_mol)
            return has_core, frag_smi, canon_mol

        # Path to keep ingthe tautomeric form of the original core
        else:
            core_mol = Chem.MolFromSmiles(core_smi)

            # Initialising the enumerator, canonicalising the molecule and obtaining all tautomeric
            # forms of the original core
            core_tauts = enumerator.Enumerate(core_mol)

            # Setting flag for whether molecules have a one of the tautomeric cores
            has_core = False

            # Checking if the mol has the original core
            if canon_mol.HasSubstructMatch(core_mol):

                # If so, return the canonical smiles, fragment and core flag
                has_core = True
                canon_smi = Chem.MolToSmiles(canon_mol)
                frag_mol = Chem.ReplaceCore(mol, core_mol)
                frag_smile = Chem.MolToSmiles(frag_mol)

                return has_core, frag_smile, canon_smi

            # If it doesnt have the original core, check the tautomeric forms
            else:
                for tautomers in core_tauts:

                    # If it has one of the tautometic forms, substitute the core with
                    # a dummy atom
                    if mol.HasSubstructMatch(tautomers):
                        has_core = True
                        frag_mol = Chem.ReplaceCore(mol, tautomers)
                        frag_smile = Chem.MolToSmiles(frag_mol)

                        # Canonicalise the fragment and specify atom positions
                        canon_frag_mol = enumerator.Canonicalize(frag_mol)
                        frag_atoms = canon_frag_mol.GetAtoms()

                        # Find the dummy atom position
                        for atom in frag_atoms:
                            if atom.GetSymbol() == "*":

                                # Find which atom was bonded to the dummy atom
                                # (which where the original core was bonded)
                                bonds = canon_frag_mol.GetAtomWithIdx(
                                    atom.GetIdx()
                                ).GetBonds()
                                dummy_pos = atom.GetIdx()

                                for bond in bonds:
                                    neighbour_pos = bond.GetOtherAtomIdx(atom.GetIdx())

                        # Remove the dummy atom
                        emol = Chem.EditableMol(canon_frag_mol)
                        emol.RemoveAtom(dummy_pos)
                        canon_frag_mol = emol.GetMol()
                        frag_smile = Chem.MolToSmiles(canon_frag_mol)

                        # Substitute the original core onto the original position
                        combined_frags = Chem.CombineMols(canon_frag_mol, core_mol)
                        emol_combined = Chem.EditableMol(combined_frags)

                        # Atom number on core where fragment was subsituted
                        sub_atom = sub_point + len(frag_atoms) - 1

                        # Add single bond
                        emol_combined.AddBond(
                            neighbour_pos, sub_atom, Chem.rdchem.BondType.SINGLE
                        )
                        final_mol = emol_combined.GetMol()

                        # Remove the remaining H atoms (required)
                        canon_mol = Chem.RemoveHs(final_mol)
                        canon_smi = Chem.MolToSmiles(canon_mol)

                        return has_core, frag_smile, canon_smi

                # If it still doesnt have the core just return the canonicalised SMILES
                # string without the core
                if not has_core:
                    frag_smile = None
                    uncanon_smi = Chem.MolToSmiles(mol)
                    return has_core, frag_smile, uncanon_smi

    def _adjust4pH(self, smi: str, ph: float = 7.4, phmodel: str = "OpenEye"):
        """
        Description
        -----------
        Function to adjust smiles strings for a defined pH value

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

    def _kekuliseSmiles(self, mol):
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def _loadData(
        self,
        mol_dir: str,
        filename: str,
        mol_type: str,
        prefix: str = "HW-",
        retain_ids: bool = False,
        pymolgen: bool = True,
        core: str = "Cc1noc(C)c1-c1ccccc1",
        sub_point: int = None,
        keep_core: bool = True,
    ):
        """
        Description
        -----------
        Function which uses the _process_inchi & _adjust4ph functions to obtain canonical smiles from
        different molecular generation outputs. Currently only uses PyMolGen outputs

        Parameters
        ----------
        mol_dir (str)       Path to directory where molecule files are kept
        filename (str)      Name of file containing the molecules
        prefix (str)        Prefix for the dataset
        pymolgen (bool)     Flag to say the molecule files are formatted by PyMolGen
        core (str)          Original core to use in the molecule canonicalisation (_process_inchi() function)
        sub_point (int)     Point where fragments are substituted on previously defined core for the
                            molecule canonicalisation (_process_inchi() function)

        Returns
        -------
        A pd.DataFrame of the following format:
         ____ _____ _____________ ________ _________
        | ID | Mol | Frag_SMILES | SMILES | Kek SMI |
        |____|_____|_____________|________|_________|
        | id | mol |     frag    |  smi   | kek smi |
        |____|_____|_____________|________|_________|
        """

        # If molecules come from PyMolGen then carry out this processing
        if pymolgen:
            with open(f"{mol_dir}/{filename}", "r") as file:
                lines = file.readlines()

            # Obtain the InChI list from the exp.inchi files
            inchi_list = [line.strip() for line in lines]

            # Process the InChIs into their canonical forms
            canon_mol_ls, frag_smi_ls, canon_smi_ls, kek_smi_ls = self._process_inchi(
                mol_type=mol_type,
                mol_list=inchi_list,
                sub_point=sub_point,
                core=core,
                keep_core=keep_core,
            )

        else:
            mol_df = pd.read_csv(f"{mol_dir}/{filename}")
            smi_ls = mol_df["SMILES"].tolist()

            # Process the InChIs into their canonical forms
            canon_mol_ls, frag_smi_ls, canon_smi_ls, kek_smi_ls = self._process_inchi(
                mol_type=mol_type,
                mol_list=smi_ls,
                sub_point=sub_point,
                core=core,
                keep_core=keep_core,
            )

            # Adjust protonation state of canonical molecule at given pH

        ph_canon_smi_ls = [
            self._adjust4ph(smi, phmodel="OpenEye") for smi in canon_smi_ls
        ]

        id_ls = (
            [f"{prefix}{i+1}" for i in range(len(canon_mol_ls))]
            if not retain_ids
            else list(mol_df["ID"])
        )

        # Format the data prior to pd.DataFrame entry
        data = {
            "ID": id_ls,
            "Mol": canon_mol_ls,
            "Frag_SMILES": frag_smi_ls,
            "SMILES": ph_canon_smi_ls,
            "Kekulised_SMILES": kek_smi_ls,
        }

        self.smi_df = pd.DataFrame(data)
        self.lilly_smi_df = self.apply_lilly_rules(self.smi_df)

        return self.lilly_smi_df

    def _calculateLogPoe(self, smi: str):
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
            self.logger.error("ERROR: {}".format(smi))
        else:
            # Calculate logP
            try:
                logp = oe.OEGetXLogP(mol, atomxlogps=None)
            except RuntimeError as e:
                self.logger.error(f"Runtime error for: {smi}\n{e}")
        return logp

    def _getDesc(self, mol, missingVal=None, descriptor_set: str = "RDKit"):
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
                except Exception as e:
                    self.logger.error(f"Error calculating descriptors: {e}")
                    val = missingVal
                res[name] = val

        elif descriptor_set == "Mordred":
            calc = Calculator(descriptors, ignore_3D=True)

            try:
                desc = calc(mol)

                for name, value in desc.items():
                    res[str(name)] = float(value) if value is not None else missingVal

            except Exception as e:
                self.logger.error(f"Error calculating descriptors: {e}")
                for descriptor in calc.descriptors:
                    res[str(descriptor)] = missingVal

        return res

    def _calcDescDF(
        self, df: pd.DataFrame = None, descriptor_set: str = "RDKit"
    ):
        """
        Description
        -----------
        Function to calculate descriptors for a whole dataset

        Parameters
        ----------
        df (pd.DataFrame)       pd.DataFrame you want to calculate descriptors for, must have
                                column named 'Mol' containing RDKit mol objects
        descriptor_set (str)    Choose the descriptor set you want to generate in the (_getDesc() function)
                                either 'RDKit' or 'Mordred'

        Returns
        -------
        DataFrame containing all SMILES and their descriptors in the following format:

         ____ _____ _____________ ________ _________ ______ _______ _____
        | ID | Mol | Frag_SMILES | SMILES | Kek SMI |desc1 | desc2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|
        | id | mol |     frag    |  smi   | kek smi |val 1 |  val2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|

        """

        # Setting up temporary df so to not save over self.smi_df
        if df is not None:
            tmp_df = df
        else:
            tmp_df = self.lilly_smi_df.copy()

        # Getting the descriptors for each mol object and saving the dictionary
        # in a column named descriptors
        tmp_df["Descriptors"] = tmp_df["Mol"].apply(
            self._getDesc, args=(None, descriptor_set)
        )

        # Making a new pd.Dataframe with each descriptor as a column, setting the
        # index to match self.smi_df (or tmp_df)
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

        # Concatenating the two dfs to give the full set of descriptors and SMILES
        self.final_df = pd.concat([tmp_df, desc_df], axis=1, join="inner").drop(
            columns=["Descriptors"]
        )

        if "nARing" in self.final_df.columns:
            self.final_df.rename(columns={"nARing": "NumAromaticRings"}, inplace=True)
        if "MW" in self.final_df.columns:
            self.final_df.rename(columns={"MW": "MolWt"}, inplace=True)

        self.final_df["oe_logp"] = self.final_df["SMILES"].apply(
            self._calculateLogPoe
        )
        self.final_df["PFI"] = (
            self.final_df["NumAromaticRings"] + self.final_df["oe_logp"]
        )

        return self.final_df

    def df2String(self, df: pd.DataFrame, **kwargs):
        """
        Description
        -----------
        Converta a Data Frame to a string

        Parameters
        ----------
        **kwargs    Arguments for the pd.DataFrame.to_csv() function

        Returns
        -------
        String representation of a dataframe

        """

        string = StringIO()
        df.to_csv(string, **kwargs)
        return string.getvalue()


    def _filterDF(
        self,
        mw_budget: int = 600,
        n_arom_rings_limit: int = 3,
        PFI_limit: int = 8,
        remove_3_membered_rings: bool = True,
        remove_4_membered_rings: bool = True,
        max_fused_ring_count: int = 1,
        pass_lilly_rules: bool = True,
    ):
        """
        Description
        -----------
        Function to filter undesirable molecules from the dataset

        Parameters
        ----------
        mw_budget (int)                 Setting a molecular weight budget for molecules
        n_arom_rings_limit (int)        Setting a limit for the number of aromatic rings for molecule
        PFI_limit (int)                 Setting a PFI limit for molecules (need to implement after
                                        OpenEye License comes)
        remove_*_membered_rings (bool)  Flag to remove 3 or 4 membered cycles
        pass_lilly_rules (bool)         Flag to check if molecules pass the LillyMedChemRules

        Returns
        -------
        A pd.DataFrame of the same format as the _calcDescDF output, but with molecules
        filtered off with the specified filters

        """

        # Obtaining all molecules which pass the defined filters
        all_passing_mols = self.final_df[
            (self.final_df["MolWt"] <= mw_budget)
            & (self.final_df["NumAromaticRings"] <= n_arom_rings_limit)
            &
            # (batch_df['PFI'] <= PFI_limit) &
            (self.final_df["Lilly_rules_pass"] == pass_lilly_rules)
        ]
        # (batch_df['max_fused_rings'] == max_fused_ring_count)

        filtered_smi = []

        for index, rows in all_passing_mols.iterrows():
            for mol in rows["Mol"].GetRingInfo().AtomRings():
                if (remove_3_membered_rings and len(mol) == 3) or (
                    remove_4_membered_rings and len(mol) == 4
                ):
                    filtered_smi.append(rows["SMILES"])

        self.filtered_results = all_passing_mols[
            ~self.final_df["SMILES"].isin(filtered_smi)
        ]

        columns_to_drop = ["Mol"]

        # for column in self.filtered_results.columns:
        #     if column == 'Lilly_rules_pass':
        #         continue
        #     try:
        #         if len(np.unique(self.filtered_results[column].values)) == 1:
        #             columns_to_drop.append(column)
        #     except TypeError:
        #         continue

        self.filtered_results.drop(columns=columns_to_drop, inplace=True)
        return self.filtered_results

    def _make_chunks(
        self,
        df: pd.DataFrame,
        chunksize: int,
        save_data: bool = False,
        save_path: str = None,
        filename: str = None,
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

        Returns
        -------
        Print statements to show which chunk is being saved and where, and a list of the chunks
        """

        chunks = [df.iloc[i : i + chunksize] for i in range(0, df.shape[0], chunksize)]

        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Saving chunk {i} to:\n{save_path}{filename}")
            if save_data:
                chunk.to_csv(
                    f"{save_path}{filename}_{i+1}.csv.gz",
                    compression="gzip",
                    index="ID",
                )

        return chunks

    def FormatData(
        self,
        mol_dir: str,
        mol_filename: str,
        mol_type: str,
        prefix: str,
        retain_ids: bool,
        pymolgen: bool,
        save_data: bool,
        save_path: str = None,
        save_chunk_filename: str = None,
        chunksize: int = 10000,
        core: str = "Cc1noc(C)c1-c1ccccc1",
        sub_point: int = None,
        keep_core: bool = True,
        descriptor_set: str = "RDKit",
    ):
        """
        Description
        -----------
        Function to load, canonicalise, filter and format the data in a oner

        Parameters
        ----------
        mol_dir (str)                   Path to directory where molecule files are kept
        mol_filename (str)              Name of file containing the molecules
        prefix (str)                    Prefix for the dataset
        pymolgen (bool)                 Flag to say the molecule files are formatted by PyMolGen
        save_data (bool)                Flag to save the chunks
        save_path (str)                 Path to save the chunks to
        save_chunk_filename (str)       Name to save chunks as, function will number them for you
        chunksize (int)                 Number of rows you want in each chunk of your Data Frame
        core (str)                      Original core to use in the molecule canonicalisation (_process_inchi() function)
        sub_point (int)                 Point where fragments are substituted on previously defined core for the
                                        molecule canonicalisation (_process_inchi() function)
        keep_core (bool)                Flag to keep original tautomer of core in the structure
        descriptor_set (str)            Descriptor set used as the descriptors
        """

        self._loadData(
            mol_dir=mol_dir,
            mol_type=mol_type,
            filename=mol_filename,
            prefix=prefix,
            retain_ids=retain_ids,
            pymolgen=pymolgen,
            core=core,
            sub_point=sub_point,
            keep_core=keep_core,
        )
        self.logger.info("Data Loaded")

        self._calcDescDF(df=self.lilly_smi_df, descriptor_set=descriptor_set)

        self.logger.info("Descriptors Calculates")

        self._filterDF()

        self.logger.info("Data Frame Filtered")

        self._make_chunks(
            df=self.filtered_results,
            chunksize=chunksize,
            save_data=save_data,
            save_path=save_path,
            filename=save_chunk_filename,
        )


class mp_Dataset_Formatter:
    def __init__(self, n_processes: int = None):

        if n_processes is None:
            self.cpu_count = cpu_count()
        else:
            self.cpu_count = 2

    def _process_line(self, line):
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
        self.logger.info("Making Chunks")

        chunks = []

        if pymolgen:
            with open(f"{mol_dir}/{filename}", "r") as file:
                lines = file.readlines()

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
                self.logger.info(f"Chunk {chunk_idx + 1} made ({len(chunk_df)} lines)...")

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

        self.logger.info(f"Made {len(chunks)} chunks")

        return chunks

    def _make_chunks_wrapper(self, args):
        return self._make_chunks(*args)

    def _process_mols(
        self,
        mol_type: str,
        mol_list: str,
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
        3 lists:
                canon_mol_list = List of canonical RDKit mol objects
                frag_smi_list  = List of the canonical fragments added onto the original core
                                (only works if keep_core is True, core and sub point are defined)
                canon_smi_list = List of canonical SMILES strings, determined by RDKit
        """

        # Converting all inchis into RDKit mol objects

        if mol_type == "inchi":
            mol_list = [
                Chem.MolFromInchi(x)
                for x in mol_list
                if Chem.MolFromInchi(x) is not None
            ]
        elif mol_type == "smiles":
            mol_list = [
                Chem.MolFromSmiles(x)
                for x in mol_list
                if Chem.MolFromSmiles(x) is not None
            ]

        # Canonicalising the SMILES and obtaining the 3 lists produced by the _canonicaliseSmiles() function
        results = [
            self._canonicaliseSmiles(
                mol, keep_core=keep_core, core_smi=core, sub_point=sub_point
            )
            for mol in mol_list
        ]

        # Isolating each output list
        frag_smi_list = [result[1] for result in results if results[0]]
        canon_smi_list = [result[2] for result in results if results[0]]
        canon_mol_list = [result[3] for result in results if results[0]]
        kekulised_smi_ls = [self._kekuliseSmiles(mol) for mol in canon_mol_list]

        return canon_mol_list, frag_smi_list, canon_smi_list, kekulised_smi_ls

    def _processMolsWrapper(self, args):
        return self._process_mols(*args)

    def _canonicaliseSmiles(
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
        canon_smi (str)             SMILES string of the canonical form of the molecule after added onto
                                    sub_core
        frag_smi (str)              SMILES string of the fragment added onto the core
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

    def _adjust4ph(self, smi: str, ph: float = 7.4, phmodel: str = "OpenEye"):
        """
        Description
        -----------
        Function to adjust smiles strings for a defined pH value

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

    def _kekuliseSmiles(self, mol):
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def _df2String(self, df: pd.DataFrame, **kwargs):
        """
        Description
        -----------
        Converta a Data Frame to a string

        Parameters
        ----------
        **kwargs    Arguments for the pd.DataFrame.to_csv() function

        Returns
        -------
        String representation of a dataframe

        """

        string = StringIO()
        df.to_csv(string, **kwargs)
        return string.getvalue()

    def applyLillyRules(
        self,
        df=None,
        smiles=[],
        smi_input_filename=None,
        cleanup=True,
        run_in_temp_dir=True,
        lilly_rules_script=str(ROOT_DIR)
        + "/Lilly-Medchem-Rules/Lilly_Medchem_Rules.rb",
    ):
        """
        Apply Lilly rules to SMILES in a list or a DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing SMILES
        smiles_col: str
            Name of SMILES column

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
            error_message = f"Cannot find Lilly rules script (Lilly_Medchem_Rules.rb) at: {lilly_rules_script_path}"
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)

        smi_file_txt = self.df2String(
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
            self.logger.warning("{}".format(lilly_results.stderr.decode("utf-8")))
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

        self.logger.debug(df[~df["ID"].isin(df_out["ID"])])

        # Check all molecules accounted for:
        if len(df_out) != len(df):
            raise ValueError(
                "Some compounds missing, {} molecules input, but {} compounds output.".format(
                    len(df), len(df_out)
                )
            )

        # df['Lilly_rules_pass'].fillna(False, inplace=True)

        return df_out.set_index("ID")

    def _loadData(
        self,
        mol_dir: str,
        filename: str,
        prefix: str,
        pymolgen: bool,
        mol_type: str,
        chunksize: int = 10000,
        retain_ids: bool = False,
        core: str = None,
        sub_point: int = None,
        keep_core: bool = False,
        save_chunks: bool = False,
        save_path: str = None,
    ):

        self.data_ls = []

        chunks = self._make_chunks(
            mol_dir, filename, retain_ids, pymolgen, prefix, chunksize
        )
        self.logger.info(f"Total chunks: {len(chunks)}")

        arguments = [
            (mol_type, chunk["SMILES"].tolist(), sub_point, core, keep_core)
            for chunk in chunks
        ]

        with Pool() as pool:
            results = pool.map(self._processMolsWrapper, arguments)

        for i, (chunk, item) in enumerate(zip(chunks, results)):
            self.logger.info(f"Processing chunk {i+1}")

            canon_mol_ls, frag_smi_ls, canon_smi_ls, kek_smi_ls = item

            ph_canon_smi_ls = [
                self._adjust4ph(smi, phmodel="OpenEye") for smi in canon_smi_ls
            ]

            data = {
                "ID": chunk.index,
                "Mol": canon_mol_ls,
                "Frag_SMILES": frag_smi_ls,
                "SMILES": ph_canon_smi_ls,
                "Kekulised_SMILES": kek_smi_ls,
            }

            smi_df = pd.DataFrame(data)
            lilly_smi_df = self._applyLillyRules(smi_df)

            self.data_ls.append(lilly_smi_df)

            if save_chunks:
                lilly_smi_df.to_csv(
                    f"{save_path}/raw_chunks_{i+1}.csv.gz",
                    index="ID",
                    compression="gzip",
                )

        return self.data_ls

    def _calculateLogPoe(self, smi: str):
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
            self.logger.warning(f"Failed to process {smi}")
        else:
            # Calculate logP
            try:
                logp = oe.OEGetXLogP(mol, atomxlogps=None)
            except RuntimeError as e:
                self.logger.error(f"Encountered error processing {smi}:\n{e}")
        return logp

    def _getDesc(self, mol, missingVal=None, descriptor_set: str = "RDKit"):
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
                except Exception as e:
                    self.logger.error(f"Failed to generate RDKit descriptor: {name}")
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
                self.logger.error(f"Failed to generate Mordred descriptor: {name}")
                for descriptor in calc.descriptors:
                    res[str(descriptor)] = missingVal

        return res

    def _getDescWrapper(self, args):

        return self._getDesc(*args)

    def _calcDescDF(
        self,
        df_list: list = None,
        descriptor_set: str = "RDKit",
        save_desc_df: bool = False,
        save_path: str = None,
        filename: str = None,
    ):
        """
        Description
        -----------
        Function to calculate descriptors for a whole dataset

        Parameters
        ----------
        df (pd.DataFrame)       pd.DataFrame you want to calculate descriptors for, must have
                                column named 'Mol' containing RDKit mol objects
        descriptor_set (str)    Choose the descriptor set you want to generate in the (_getDesc() function)
                                either 'RDKit' or 'Mordred'

        Returns
        -------
        DataFrame containing all SMILES and their descriptors in the following format:

         ____ _____ _____________ ________ _________ ______ _______ _____
        | ID | Mol | Frag_SMILES | SMILES | Kek SMI |desc1 | desc2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|
        | id | mol |     frag    |  smi   | kek smi |val 1 |  val2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|

        """

        # Setting up temporary df so to not save over self.smi_df
        if df_list is not None:
            tmp_df_ls = df_list
        else:
            tmp_df_ls = self.data_ls

        self.batch_desc_ls = []
        self.batch_full_ls = []

        for tmp_df in tmp_df_ls:
            # Getting the descriptors for each mol object and saving the dictionary
            # in a column named descriptors
            rows = tmp_df.to_dict("records")

            with Pool(processes=self.cpu_count) as pool:
                desc_mp_item = pool.map(
                    self._getDescWrapper,
                    [(row["Mol"], None, descriptor_set) for row in rows],
                )

            # Making a new pd.Dataframe with each descriptor as a column, setting the
            # index to match self.smi_df (or tmp_df)
            tmp_df["Descriptors"] = desc_mp_item

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

            self.batch_desc_ls.append(desc_df)

            # Concatenating the two dfs to give the full set of descriptors and SMILES
            batch_df = pd.concat([tmp_df, desc_df], axis=1, join="inner").drop(
                columns=["Descriptors"]
            )

            batch_df["oe_logp"] = batch_df["SMILES"].apply(self._calculateLogPoe)
            batch_df["PFI"] = batch_df["NumAromaticRings"] + batch_df["oe_logp"]

            self.batch_full_ls.append(batch_df)

        if save_desc_df:
            for i, dfs in enumerate(self.batch_desc_ls):
                dfs.to_csv(
                    save_path + filename + f"_{i+1}.csv.gz",
                    index_col="ID",
                    compression="gzip",
                )
                self.logger.debug(f"Saved {filename}_{i+1}")

        return self.batch_desc_ls, self.batch_full_ls

    def _filterDF(
        self,
        mw_budget: int = 600,
        n_arom_rings_limit: int = 3,
        PFI_limit: int = 8,
        remove_3_membered_rings: bool = True,
        remove_4_membered_rings: bool = True,
        max_fused_ring_count: int = 1,
        pass_lilly_rules: bool = True,
        chembl: bool = False,
    ):
        """
        Description
        -----------
        Function to filter undesirable molecules from the dataset

        Parameters
        ----------
        mw_budget (int)                 Setting a molecular weight budget for molecules
        n_arom_rings_limit (int)        Setting a limit for the number of aromatic rings for molecule
        PFI_limit (int)                 Setting a PFI limit for molecules (need to implement after
                                        OpenEye License comes)
        remove_*_membered_rings (bool)  Flag to remove 3 or 4 membered cycles
        pass_lilly_rules (bool)         Flag to check if molecules pass the LillyMedChemRules

        Returns
        -------
        A pd.DataFrame of the same format as the _calcDescDF output, but with molecules
        filtered off with the specified filters

        """

        self.filtered_df_ls = []
        for df in self.batch_full_ls:
            if not chembl:
                # Obtaining all molecules which pass the defined filters
                all_passing_mols = df[
                    (df["MolWt"] <= mw_budget)
                    & (df["NumAromaticRings"] <= n_arom_rings_limit)
                    & (df["PFI"] <= PFI_limit)
                    & (df["Lilly_rules_pass"] == pass_lilly_rules)
                ]

                filtered_smi = []
                for index, rows in all_passing_mols.iterrows():
                    for mol in rows["Mol"].GetRingInfo().AtomRings():
                        if (remove_3_membered_rings and len(mol) == 3) or (
                            remove_4_membered_rings and len(mol) == 4
                        ):
                            filtered_smi.append(rows["SMILES"])

                filtered_results = all_passing_mols[~df["SMILES"].isin(filtered_smi)]
            columns_to_drop = ["Mol"]

            if chembl:
                filtered_results = df

            filtered_results.drop(columns=columns_to_drop, inplace=True)
            self.filtered_df_ls.append(filtered_results)

        return self.filtered_df_ls

    def _makeFinalChunks(
        self,
        chunksize: int,
        save_full_data: bool = False,
        gen_desc_chunks: bool = False,
        save_desc_data: bool = True,
        descriptor_set: str = "RDKit",
        full_save_path: str = None,
        desc_save_path: str = None,
        filename: str = None,
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

        Returns
        -------
        Print statements to show which chunk is being saved and where, and a list of the chunks
        """

        if len(self.filtered_df_ls) == 1:
            full_df = self.filtered_df_ls[0]
        else:
            full_df = pd.concat(self.filtered_df_ls)

        full_chunks = [
            full_df.iloc[i : i + chunksize]
            for i in range(0, full_df.shape[0], chunksize)
        ]
        for i, chunk in enumerate(full_chunks):
            if save_full_data:
                self.logger.debug(f"Saving chunk {i} to:\n{full_save_path}{filename}")
                chunk.to_csv(
                    f"{full_save_path}{filename}_{i+1}.csv.gz",
                    compression="gzip",
                    index="ID",
                )

        if gen_desc_chunks:
            if descriptor_set == "RDKit":
                columns, fns = zip(*Descriptors.descList)
            if descriptor_set == "Mordred":
                calc = Calculator(descriptors, ignore_3D=True)
                desc_names = [str(desc) for desc in calc.descriptors]
                columns = []
                for desc in desc_names:
                    if desc == "naRing":
                        columns.append("NumAromaticRings")
                    elif desc == "MW":
                        columns.append("MolWt")
                    else:
                        columns.append(desc)

            full_desc = full_df.loc[:, columns]
            full_desc_chunks = [
                full_desc.iloc[i : i + chunksize]
                for i in range(0, full_df.shape[0], chunksize)
            ]

            for i, chunk in enumerate(full_desc_chunks):
                if save_desc_data:
                    self.logger.debug(f"Saving chunk {i} to:\n{desc_save_path}{filename}")
                    chunk.to_csv(
                        f"{desc_save_path}{filename}_{i+1}.csv.gz",
                        compression="gzip",
                        index="ID",
                    )

        return full_chunks, full_desc_chunks


class DatasetAccessor:
    def __init__(
        self,
        original_path: str,
        wait_time: int = 30,
        max_wait: int = 21600,
        logger=None, 
        log_level="DEBUG",
        log_path=None

    ):

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

            self.logger.info(f"Initialised logger: {log_path}")

        self.original_path = Path(original_path)
        self.wait_time = wait_time
        self.max_wait = max_wait

        # Set up file lock path and object
        self.lock_path = self.original_path.with_suffix(".lock")
        self.lock = FileLock(self.lock_path, timeout=self.max_wait)

    def getExclusiveAccess(self):
        """
        Acquire exclusive access to the file using a file lock.
        Waits and retries until access is granted or max_wait is exceeded.
        """
        waited = 0

        while waited <= self.max_wait:
            try:
                self.lock.acquire(timeout=self.wait_time)
                self.logger.debug(f"Lock acquired for: {self.original_path}")
                return str(self.original_path)

            except TimeoutError:
                self.logger.warning(
                    f"File is locked. Retrying in {self.wait_time} seconds... "
                    f"Waited: {waited}/{self.max_wait}"
                )
                time.sleep(self.wait_time)
                waited += self.wait_time

        self.logger.error(f"Timed out after {self.max_wait} seconds trying to access: {self.original_path}")
        return None

    def releaseFile(self):
        """
        Releases the file lock and removes the lock file.
        """
        if self.lock.is_locked:
            lock_file_path = Path(str(self.original_path) + ".lock")

            try:
                self.lock.release()
                self.logger.debug(f"Lock released for: {self.original_path}")

                # Delete .lock file if it exists
                if lock_file_path.exists():
                    lock_file_path.unlink()
                    self.logger.debug(f".lock file deleted: {lock_file_path}")

            except Exception as e:
                self.logger.error(f"Failed to release lock or delete .lock file: {e}")

    def editDF(
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
            df_path = self.original_path  # 🔧 fix: use original path

        if df is None:
            try:
                self.df = pd.read_csv(df_path, index_col=index_col)
            except UnicodeDecodeError:
                self.df = pd.read_csv(df_path, index_col=index_col, compression="gzip")

        if idxs_to_edit is not None and vals_to_enter is not None:
            for idx, val in zip(idxs_to_edit, vals_to_enter):
                self.df.loc[idx, column_to_edit] = str(val)

        elif data_dict is not None:
            for idx, val in data_dict.items():
                self.df.loc[idx, column_to_edit] = str(val)
        else:
            raise ValueError(
                "Either data_dict, idxs_to_edit, or vals_to_enter must be provided"
            )

        compression = "gzip" if str(df_path).endswith(".gz") else None
        self.df.to_csv(df_path, index_label=index_col, compression=compression)