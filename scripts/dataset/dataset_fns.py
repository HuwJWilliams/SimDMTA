import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import openeye as oe
from openeye import oechem
import openbabel as ob
from multiprocessing import Pool
from pathlib import Path
import tempfile
from io import StringIO
import subprocess
import glob
import time
import logging
from filelock import FileLock
from typing import Union

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}

class DatasetFormatter:

    def __init__(self, logger=None, log_level="DEBUG", log_path=FILE_DIR):

        # Setting up logging
        log_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
        self.logger = logging.getLogger(f"DatasetFormatter")
        self.logger.setLevel(log_level)

        log_file = Path(log_path) / "DatasetFormatter.log"
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

    def _convert2Mol(self, idx, smi, format):
        try:
            if format.lower() == "smiles":
                return Chem.MolFromSmiles(smi)
            
            elif format.lower() == "inchi":
                return Chem.MolFromInchi(smi)
            
        except Exception as e:
            self.logger.debug(f"Error processing {id}:\n{e}\nSetting value to None")
            return None
        
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
    
    def _canonicaliseSmiles(self, mol, keep_core:bool=False, core_smi: str=None, sub_point: int=None):
        
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)

        # If maintaining the original core is not an issue then just canonicalise as usual,
        # Does not show fragment SMILES strings
        if not keep_core:
            has_core = True
            frag_smi = None
            canon_smi = Chem.MolToSmiles(canon_mol)
            return has_core, frag_smi, canon_smi

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

    def _kekuliseSmiles(self, mol):
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    
    def _adjust4pH(self, smi:str, pH: float=7.4, pH_model: str="OpenEye"):
        # OpenEye pH model needs a pH of 7.4
        if pH_model.lower() == "openeye" and pH != 7.4:
            raise ValueError("Cannot use OpenEye pH conversion for pH != 7.4")

        # Use OpenBabel for pH conversion
        if pH_model.lower() == "openbabel":
            ob_conv = ob.OBConversion()
            ob_mol = ob.OBMol()
            ob_conv.SetInAndOutFormats("smi", "smi")
            ob_conv.ReadString(ob_mol, smi)
            ob_mol.AddHydrogens(
                False, True, pH
            )
            ph_smi = ob_conv.WriteString(ob_mol, True)  # <- Trim White Space

            # Check that pH adjusted SMILES can be read by RDKit,
            # if not return original SMILES
            if Chem.MolFromSmiles(ph_smi) is None:
                ph_smi = smi

        # Use OpenEye for pH conversion
        elif pH_model.lower() == "openeye":
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, smi)
            oe.OESetNeutralpHModel(mol)
            ph_smi = oechem.OEMolToSmiles(mol)

        return ph_smi
    
    def _applyLillyRules(
            self,
            df=None,
            smiles=[],
            smi_input_filename=None,
            cleanup=True,
            run_in_temp_dir=True,
            lilly_rules_script=FILE_DIR / "Lilly-Medchem-Rules" / "Lilly_Medchem_Rules.rb",
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

            smi_file_txt = self._df2String(
                df.reset_index()[["Kekulised_SMILES", "ID"]], sep=" ", header=False, index=False
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

            # Check all molecules accounted for:
            if len(df_out) != len(df):
                raise ValueError(
                    "Some compounds missing, {} molecules input, but {} compounds output.".format(
                        len(df), len(df_out)
                    )
                )

            # df['Lilly_rules_pass'].fillna(False, inplace=True)

            return df_out.set_index("ID")

    def _calculateLogPOE(self, smi:str):

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
    
    def _processMols(self,
                     df: Union[str, pd.DataFrame],
                     input_representation_type: str="SMILES",
                     column: str="SMILES",
                     keep_core: bool=False,
                     core_smi: str=None,
                     sub_point: int=None,
                     pH: float=7.4,
                     pH_model: str="OpenEye",
                     lilly_rules_script=FILE_DIR / "Lilly-Medchem-Rules" / "Lilly_Medchem_Rules.rb",
                      ):
        
        if isinstance(df, str):
            df = pd.read_csv(df)
        
        self.logger.debug(f"Data frame head:\n{df.head()}")

        df['Mol'] = [self._convert2Mol(idx=idx, smi=row[column], format=input_representation_type) for idx, row in df.iterrows()]
        failed_mols = df[df['Mol'].isna()].index

        if len(failed_mols) != 0:
            self.logger.debug(f"Failed to convert {len(failed_mols)} molecules:\n{failed_mols}")

        df = df[df['Mol'].notna()]

        self.logger.info("Generating canonical SMILES")
        df['Canon_SMILES'] = [
            self._canonicaliseSmiles(
                mol=mol, keep_core=keep_core, core_smi=core_smi, sub_point=sub_point
                )[-1]
            for mol in df['Mol']
        ]
        self.logger.debug(f"Printing data:\n{df}")


        self.logger.info("Generating kekulised SMILES for Lilly-MedChem-Rules")
        df["Kekulised_SMILES"] = [
            self._kekuliseSmiles(mol=mol)
            for mol in df['Mol']
        ]
        self.logger.debug(f"Printing data:\n{df}")


        self.logger.info("Generating pH adjusted canonical SMILES")
        df['pH_adjusted_Canon_SMILES'] = [
            self._adjust4pH(
                smi=smi, pH=pH, pH_model=pH_model
            )
            for smi in df['Canon_SMILES']
        ]
        self.logger.debug(f"Printing data:\n{df}")


        df = self._applyLillyRules(df=df, lilly_rules_script=lilly_rules_script)

        self.logger.info("Generating OpenEye logPs")
        df['oe_logp'] = [
                self._calculateLogPOE(smi=smi) for smi in df['pH_adjusted_Canon_SMILES']
        ]
        self.logger.debug(f"Printing data:\n{df}")

        self.preprocessed_df = df

        return self.preprocessed_df
    
    def _getDescriptors(self, mol, missingVal=None, descriptor_set: str="RDKit"):
        res = {}
        seen = set()  # Track already added descriptor names

        if descriptor_set.lower() == "rdkit":
            for name, func in Descriptors._descList:
                if name in seen:
                    self.logger.warning(f"Duplicate RDKit descriptor '{name}' skipped.")
                    continue

                try:
                    val = func(mol)
                except Exception as e:
                    self.logger.error(f"Failed to generate RDKit descriptor: {name} — {e}")
                    val = missingVal

                res[name] = val
                seen.add(name)

        elif descriptor_set.lower() == "mordred":
            calc = Calculator(descriptors, ignore_3D=True)

            try:
                desc = calc(mol)
                for name, value in desc.items():
                    name_str = str(name)
                    if name_str in seen:
                        self.logger.warning(f"Duplicate Mordred descriptor '{name_str}' skipped.")
                        continue
                    res[name_str] = float(value) if value is not None else missingVal
                    seen.add(name_str)

            except Exception as e:
                self.logger.error(f"Failed to generate Mordred descriptors: {e}")
                for descriptor in calc.descriptors:
                    name_str = str(descriptor)
                    if name_str not in seen:
                        res[name_str] = missingVal
                        seen.add(name_str)

        else:
            self.logger.error("Unsupported descriptor set specified. Supported: [RDKit, Mordred]")

        return res
    
    def _getDescWrapper(self, args):
        return self._getDescriptors(*args)
    
    def _calculateDescriptors(self, 
                            df: Union[str, pd.DataFrame], 
                            descriptor_set: str = "RDKit", 
                            use_mp: bool = False,
                            smi_column: str = "pH_adjusted_Canon_SMILES",
                            n_cpus: int = 1):
        # Load dataframe if file path is given
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, list):
            raise ValueError("Expected a single DataFrame, not a list.")

        df = df.copy()

        # Use 'ID' if available, else fallback to index
        ids = df.get("ID", df.index)

        # Generate Mol objects if not already present
        if "Mol" not in df.columns:
            df["Mol"] = [Chem.MolFromSmiles(smi) for smi in df[smi_column]]

        # Prepare data for descriptor calculation
        rows = df.to_dict("records")

        if use_mp:
            with Pool(processes=n_cpus) as pool:
                desc_list = pool.map(
                    self._getDescWrapper,
                    [(row["Mol"], None, descriptor_set) for row in rows]
                )
        else:
            desc_list = [self._getDescWrapper((row["Mol"], None, descriptor_set)) for row in rows]

        df["Descriptors"] = desc_list

        # Build descriptor DataFrame
        if descriptor_set.lower() == "rdkit":
            desc_df = pd.DataFrame(desc_list, columns=[name for name, _ in Descriptors.descList])
        elif descriptor_set.lower() == "mordred":
            calc = Calculator(descriptors, ignore_3D=True)
            desc_df = pd.DataFrame(desc_list, columns=[str(d) for d in calc.descriptors])
        else:
            raise ValueError("Unsupported descriptor set. Choose 'RDKit' or 'Mordred'.")

        # Fix column naming inconsistencies
        desc_df.rename(columns={"naRing": "NumAromaticRings", "MW": "MolWt"}, inplace=True)
        desc_df = desc_df.loc[:, ~desc_df.columns.duplicated()]
        desc_df.index = ids

        # Merge descriptor columns with original dataframe
        desc_df_to_merge = desc_df.loc[:, ~desc_df.columns.isin(df.columns)]
        batch_df = pd.concat([df.reset_index(drop=True), desc_df_to_merge.reset_index(drop=True)], axis=1)
        batch_df = batch_df.loc[:, ~batch_df.columns.duplicated()]
        batch_df.index = ids

        # Add calculated LogP and PFI
        batch_df["oe_logp"] = batch_df[smi_column].apply(self._calculateLogPOE)
        batch_df["PFI"] = batch_df["NumAromaticRings"] + batch_df["oe_logp"]

        return desc_df, batch_df

    
    def _makeDockingCSV(self, 
                        df: pd.DataFrame,
                        docking_column: str = "Affinity(kcal/mol)", 
                        smi_col: str = "pH_adjusted_Canon_SMILES") -> pd.DataFrame:
        """
        Generate a docking CSV file with SMILES and empty docking column.
        """
        docking_df = df[[smi_col]].copy()
        docking_df[docking_column] = ""
        docking_df.rename(columns={smi_col: "SMILES"}, inplace=True)
        return docking_df
        
    def _chunkDataFrame(self, df, chunk_size=10000):
        """Yield successive chunk_size-sized chunks from DataFrame."""
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size].copy()

    def generateDatasets(
        self,
        df: Union[str, pd.DataFrame],
        chunk_size: int = 10000,
        descriptor_set: str = "RDKit",
        save_data: bool = True,
        save_path: Union[str, Path] = FILE_DIR,
        filename_prefix: str = "dataset",
        smi_column: str = "pH_adjusted_Canon_SMILES",
        use_mp: bool = False,
        n_cpus: int = 1,
    ):
        if isinstance(df, str):
            df = pd.read_csv(df)

        if len(df) > chunk_size:
            self.logger.info(f"Splitting dataset into chunks of {chunk_size}")
            df_chunks = list(self._chunk_dataframe(df, chunk_size=chunk_size))
        else:
            df_chunks = [df]

        all_descs, all_full, all_docking = [], [], []

        for i, chunk in enumerate(df_chunks):
            self.logger.info(f"Processing chunk {i + 1}/{len(df_chunks)}")

            # Molecule preprocessing
            processed_df = self._processMols(df=chunk)

            # Descriptor generation
            desc_df, full_df = self._calculateDescriptors(
                df=processed_df,
                descriptor_set=descriptor_set,
                smi_column=smi_column,
                use_mp=use_mp,
                n_cpus=n_cpus
            )
            # Make docking CSV
            docking_df = self._makeDockingCSV(
                df=full_df,
                smi_col=smi_column,
            )

            if save_data:
                # Save descriptor file
                desc_df.to_csv(
                    Path(save_path) / f"{filename_prefix}_desc_chunk_{i+1}.csv.gz",
                    index_label="ID",
                    compression="gzip",
                )

                # Save full dataset file
                full_df.to_csv(
                    Path(save_path) / f"{filename_prefix}_full_chunk_{i+1}.csv.gz",
                    index_label="ID",
                    compression="gzip",
                )

                # Save docking CSV
                docking_df.to_csv(
                    Path(save_path) / f"{filename_prefix}_dock_chunk_{i+1}.csv",
                    index_label="ID"
                )

                self.logger.info(f"Saved all files for chunk {i + 1}")

            # Collect all outputs
            all_descs.append(desc_df)
            all_full.append(full_df)
            all_docking.append(docking_df)

        return all_descs, all_full, all_docking


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