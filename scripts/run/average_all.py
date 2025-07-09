import pandas as pd
from pathlib import Path
import numpy as np
import sys
from glob import glob
import json
import joblib
from scipy.stats import pearsonr


PROJ_DIR = Path(__file__).parent.parent.parent
FILE_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR) + "/scripts/misc/")
sys.path.insert(0, str(PROJ_DIR) + "/scripts/models/")


from misc_functions import (
    count_number_iters,
    get_chembl_molid_smi,
    get_sel_mols_between_iters,
    molid_ls_to_smiles,
)

from RF_class import PredictNewTestSet


class AverageAll:

    def __init__(
        self,
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        docking_column: str = "Affinity(kcal/mol)",
        preds_column: str = "pred_Affinity(kcal/mol)",
    ):

        self.results_dir = results_dir
        self.docking_column = docking_column
        self.preds_column = preds_column

    def _avg_stats(self, it: int, all_exp_dirs: list):

        ho_stats = pd.DataFrame()
        int_stats = pd.DataFrame()
        chembl_int_stats = pd.DataFrame()
        tr_ho_stats = pd.DataFrame()

        path = f"{PROJ_DIR}/datasets/held_out_data/"
        ft = path + 'PMG_held_out_desc_top.csv'
        tg = path + 'PMG_held_out_targ_top.csv'
        fl = path + 'PMG_rdkit_full_top.csv'

        for dir in all_exp_dirs:
            working_dir = str(dir) + f"/it{it}"

            # Load internal performance json
            try:
                with open(working_dir + "/performance_stats.json", "r") as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                int_stats = pd.concat([int_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)

            # Load hold out performance json
            try:
                with open(
                    working_dir + "/held_out_test/held_out_stats.json", "r"
                ) as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                ho_stats = pd.concat([ho_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)

            # Load trimmed hold out performance json
            working_path = Path(working_dir)
            trimmed_dir_path = working_path / "trimmed_held_out_test"
            trimmed_stats_path = trimmed_dir_path / "trimmed_held_out_stats.json"

            if not trimmed_dir_path.exists():
                try:
                    PredictNewTestSet(
                        feats=ft,
                        targs=tg,
                        full_data=fl,
                        test_set_name = 'trimmed_held_out',
                        experiment_ls=[working_path.parent.name],
                        results_dir=self.results_dir,
                        docking_column=self.docking_column
                            )
                except Exception as e:
                    print(f"Failed to recalculate trimmed test set: {e}")
                
            else:
                try:
                    with open(
                        trimmed_stats_path, "r"
                    ) as file:
                        loaded_dict = json.load(file)
                    loaded_df = pd.DataFrame([loaded_dict])
                    tr_ho_stats = pd.concat([tr_ho_stats, loaded_df], axis=0)

                except Exception as e:
                    print(e)

            # Load ChEMBL internal performance json
            try:
                with open(working_dir + "/chembl_performance_stats.json", "r") as file:
                    loaded_dict = json.load(file)
                loaded_df = pd.DataFrame([loaded_dict])
                chembl_int_stats = pd.concat([chembl_int_stats, loaded_df], axis=0)

            except Exception as e:
                print(e)

        # print(f"Internal Statistics:\n{int_stats}\n")
        # print(f"Hold Out Statistics:\n{ho_stats}\n")
        # print(f"ChEMBL Internal Statistics:\n{chembl_int_stats}\n")
                
        # Convert all data to a dictionary
        avg_int_dict = int_stats.mean().to_dict()
        avg_ho_dict = ho_stats.mean().to_dict()
        avg_chembl_int_dict = chembl_int_stats.mean().to_dict()
        avg_tr_ho_dict = tr_ho_stats.mean().to_dict()

        return avg_int_dict, avg_ho_dict, avg_chembl_int_dict, avg_tr_ho_dict

    def _avg_feat_importance(self, it: int, all_exp_dirs: list, feats_path: Path = None):

        avg_feat_df = pd.DataFrame()

        for dir in all_exp_dirs:
            working_dir = Path(dir) / f"it{it}"
            fi_path = working_dir / "feature_importance_df.csv"
            model_path = working_dir / "final_model.pkl"

            # 🚧 Generate feature_importance_df.csv if missing
            if not fi_path.exists():
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)

                        # Get feature names
                        if hasattr(model, "feature_names_in_"):
                            feature_names = model.feature_names_in_
                        elif feats_path and feats_path.exists():
                            feat_df = pd.read_csv(feats_path, index_col="ID")
                            feature_names = feat_df.columns
                        else:
                            raise ValueError(
                                f"Cannot get feature names for {working_dir}. "
                                f"Use sklearn >=1.0 or provide feats_path."
                            )

                        importance_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": model.feature_importances_
                        })
                        importance_df.to_csv(fi_path, index=False)
                        print(f"Created feature_importance_df.csv in {working_dir}")
                    except Exception as e:
                        print(f"Failed to generate feature importances in {working_dir}: {e}")
                else:
                    print(f"Model file missing: {model_path}")

            # 🔄 Now load and average the CSVs
            try:
                loaded_df = pd.read_csv(fi_path).sort_index(ascending=True)
            except Exception as e:
                print(f"Could not read {fi_path}: {e}")
                continue

            if avg_feat_df.empty:
                avg_feat_df = loaded_df
            else:
                merged_df = pd.merge(
                    avg_feat_df,
                    loaded_df,
                    on="Feature",
                    suffixes=("_df1", "_df2"),
                )
                merged_df["Importance"] = merged_df[
                    ["Importance_df1", "Importance_df2"]
                ].mean(axis=1)

                avg_feat_df = merged_df[["Feature", "Importance"]]
                avg_feat_df = avg_feat_df.sort_values(by="Feature")

        return avg_feat_df
    
    def _avg_predictions(self, it: int, all_exp_dirs:list, average_exp_dir: str):
        
        all_preds_file_ls = glob(f"{all_exp_dirs[0]}/it{it}/all_preds*")

        for preds_filename in all_preds_file_ls:
            preds_file = Path(preds_filename).name

            all_preds_df = pd.DataFrame()
            ho_df = pd.DataFrame()

            files_processed = []

            for dir in all_exp_dirs:
                working_dir = str(dir) + f"/it{it}"
                try:
                    working_csv = working_dir + "/" + preds_file
                    working_df = pd.read_csv(working_csv, index_col='ID')
                    all_preds_df = pd.concat([all_preds_df, working_df])
                    files_processed.append(working_csv)

                except Exception as e:
                    print(f"An error occurred when averaging predictions:\n{e}")
                
                
            if not all_preds_df.empty:
                avg_df = all_preds_df.groupby(all_preds_df.index).mean()

                avg_save_path = f"{average_exp_dir}/{preds_file}"
                avg_df.to_csv(avg_save_path, index_label='ID')
                print(f"Created average {preds_file}")
                #print(f"Files processed:\n{files_processed}")


        for dir in all_exp_dirs:
            working_dir = str(dir) + f"/it{it}"
                
            try:
                # Use Path chaining instead of string concat
                working_csv = Path(working_dir) / "held_out_test" / "held_out_test_preds.csv"

                # Fall back if that file doesn't exist
                if not working_csv.exists():
                    working_csv = Path(working_dir) / "held_out_test" / "held_out_preds.csv"

                # Load and concatenate
                working_df = pd.read_csv(working_csv, index_col='ID')
                ho_df = pd.concat([ho_df, working_df])

            except Exception as e:
                print(f"An error occurred when averaging hold out predictions:\n{e}")

        if not ho_df.empty:
            ho_avg_df = ho_df.groupby(ho_df.index).mean()
            ho_avg_save_path = f"{average_exp_dir}/held_out_test/held_out_test_preds.csv"
            ho_avg_df.to_csv(ho_avg_save_path, index_label='ID')


        return avg_df



    def save_json(self,
                  path: Path,
                  data: dict) -> None:
        """Helper method to save JSON data with consistent formatting."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _average_experiment(
        self, exp_suffix: str, n_iters: int, dir_prefix: str = "average_", results_dir: str=None
    ):

        results_dir = results_dir or self.results_dir
        results_path = Path(results_dir)

        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        all_exp_dirs = [
            exp for exp in results_path.iterdir()
            if exp.is_dir() 
            and exp.name.endswith(exp_suffix)
            and not exp.name.startswith("average")
        ]

        if not all_exp_dirs:
            raise ValueError(f"No experiment directories found matching suffix: {exp_suffix}")
        
        print(f"Taking averages over experiments:\n")
        for dir in all_exp_dirs:
            print(dir.name)

        # make average dir
        dir_path = Path(f"{results_dir}/{dir_prefix}{exp_suffix}")

        dir_path.mkdir(parents=True, exist_ok=True)

        for it in range(0, n_iters + 1):

            print(f"\n{'='*20} Iteration: {it} {'='*20}")

            # Create iteration directories
            working_dir = dir_path / f"it{it}/"
            working_dir.mkdir(parents=True, exist_ok=True)
            held_out_dir = working_dir / "held_out_test"
            held_out_dir.mkdir(parents=True, exist_ok=True)
            tr_held_out_dir = working_dir / "trimmed_held_out_test"
            tr_held_out_dir.mkdir(parents=True, exist_ok=True)

            # Averaging Statistics
            try:
                avg_internal, avg_held_out, avg_chembl, avg_tr_ho= self._avg_stats(
                    it=it, all_exp_dirs=all_exp_dirs
                )

                # Save performance statistics
                self.save_json(working_dir / "performance_stats.json", avg_internal)
                self.save_json(held_out_dir / "held_out_stats.json", avg_held_out)
                self.save_json(working_dir / "chembl_performance_stats.json", avg_chembl)
                self.save_json(tr_held_out_dir / "trimmed_held_out_stats.json", avg_tr_ho)

                print("Saved performance statistics")

                # Averaging Feature Importance
                avg_feat_df = self._avg_feat_importance(it=it, all_exp_dirs=all_exp_dirs)
                avg_feat_df.to_csv(f"{working_dir}/feature_importance_df.csv")

                # Averaging predictions
                avg_preds_df = self._avg_predictions(it=it, all_exp_dirs=all_exp_dirs, average_exp_dir=working_dir)
            
            except Exception as e:
                print(f"Error processing iteration {it}: {str(e)}")
                continue
            
        return dir_path


    def AverageAllExp(self,
                   exp_suffix_ls: list=["10_r",
                                        "10_mu",
                                        "10_mp",
                                        "10_mpo",
                                        "10_rmpo",
                                        "10_rmp",
                                        "10_rmu"]):
        """
        Description
        -----------
        Function to do _average_experiment() function on all experiments
        
        Parameters
        ----------
        
        Returns
        -------
        
        """

        for exp_suffix in exp_suffix_ls:
            print(f"Collecting average results for:     {exp_suffix}")
            if exp_suffix.startswith("10"):
                results_dir = self.results_dir + 'finished_results/10_mol_sel/'
                n_iters = 150
            elif exp_suffix.startswith("50"):
                results_dir = self.results_dir + 'finished_results/50_mol_sel/'
                n_iters = 30

            self._average_experiment(exp_suffix=exp_suffix, n_iters=n_iters, results_dir=results_dir)

            print("Completed\n")

        return

