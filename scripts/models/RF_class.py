import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import joblib
from joblib import Parallel, delayed
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import sys
from pathlib import Path
import shutil
import logging
import inspect

import warnings
warnings.filterwarnings('ignore')

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, f"{str(PROJ_DIR)}/scripts/misc/")
from misc_functions import countIterations


def predictNewTestSet(
                     feats: str,
                     targs: str,
                     full_data: str,
                     test_set_name: str,
                     experiment_ls: list,
                     results_dir: str=f"{str(PROJ_DIR)}/results/",
                     docking_column: str = 'Affinity(kcal/mol)',
                     logger=None,
                     ):
    """
    Description
    -----------
    Test all trained models on a new test set

    Parameters
    ----------
    feats (str)             Path to features .csv to predict, make sure ID in file
    targs (str)             Path to targets .csv to predict, make sure ID in file
    experiment_ls (list)    List of experiments to do predictions with

    Returns
    -------
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    rf_class = RFModel(docking_column=docking_column)

    feat_df = pd.read_csv(feats, index_col='ID')
    targ_df = pd.read_csv(targs, index_col='ID')
    if isinstance(targ_df, pd.DataFrame):
        targ_df = targ_df.iloc[:, 0]  # Convert to Series but keep index

    for exp in experiment_ls:
        logger.info(f"Running {exp}")
        n_iters = countIterations(results_dir + exp)
        for it in range(0, n_iters + 1):
            it_dir = results_dir + exp + f'/it{it}/'
            model_fpath = it_dir + 'final_model.pkl'
            model = joblib.load(model_fpath)

            
            preds_dir = Path(it_dir + f"{test_set_name}_test/")
            
            # Check if the path exists (could be a file or a directory)
            if preds_dir.exists():
                if preds_dir.is_file():
                    preds_dir.unlink()
                elif preds_dir.is_dir():
                    shutil.rmtree(preds_dir)
                    
            preds_dir.mkdir(parents=True, exist_ok=True)

            # Run prediction using the wrapper (includes MPO + Uncertainty)
            pred_df = rf_class.predict(
                feats=feat_df,
                save_preds=False,
                final_rf=model,
                pred_col_name=f"pred_{docking_column}",
                calc_mpo=True,
                full_data_fpath=full_data
            )

            # Save prediction DataFrame (includes MPO and Uncertainty)
            pred_df.to_csv(f"{preds_dir}/{test_set_name}_preds.csv", index_label="ID")

            # Calculate performance metrics using true vs predicted
            true_vals = targ_df.loc[pred_df.index].astype(float)
            pred_vals = pred_df[f"pred_{docking_column}"].astype(float)
            logger.debug(true_vals.head(10))
            logger.debug(pred_vals.head(10))
            errors = true_vals - pred_vals

            bias = np.mean(errors)
            sdep = (np.mean((true_vals - pred_vals - (np.mean(true_vals - pred_vals))) ** 2)) ** 0.5
            mse = mean_squared_error(true_vals, pred_vals)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_vals, pred_vals)
            r_pearson, p_pearson = pearsonr(true_vals, pred_vals)

            # Save performance stats
            performance_dict = {
                "Bias": round(float(bias), 4),
                "SDEP": round(float(sdep), 4),
                "MSE": round(float(mse), 4),
                "RMSE": round(float(rmse), 4),
                "r2": round(float(r2), 4),
                "pearson_r": round(float(r_pearson), 4),
                "pearson_p": round(float(p_pearson), 4),
            }
                  
            with open(f"{preds_dir}/{test_set_name}_stats.json", "w") as file:
                json.dump(performance_dict, file, indent=4)

    return


class RFModel:
    def __init__(self, docking_column: str, logger=None, log_level="DEBUG", log_path=None,
                 cv_class=None, cv_kwargs: dict = None):
        """
        Description
        -----------
        Initialising the ML models class

        Parameters
        ----------
        docking_column : str
            column name which docking scores are saved under
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

            self.logger.info(f"Initialised logger: {log_path}")
        
        self.docking_column = docking_column
        self.cv_class = cv_class if cv_class else KFold
        self.cv_kwargs = cv_kwargs if cv_kwargs else {}


    def _setInnerCV(self, n_splits: int = 5, set_seed: int = None, **cv_kwargs):
        seed = rand.randint(0, 2**31) if set_seed is None else set_seed
        self.logger.debug(f"Random seed: {seed}")
        self.logger.debug(f"Cross-Validation class: {self.cv_class}")

        kwargs = self.cv_kwargs.copy()
        cv_params = inspect.signature(self.cv_class).parameters

        if 'n_splits' in cv_params:
            kwargs.setdefault('n_splits', n_splits)

        # If the CV class supports shuffle and random_state, enable shuffling to allow reproducibility
        if 'shuffle' in cv_params:
            kwargs['shuffle'] = True  # Force it True if using random_state

        if 'random_state' in cv_params:
            kwargs['random_state'] = seed

        self.inner_cv = self.cv_class(**kwargs)
        self.logger.debug(f"CV initialised with params: {kwargs}")
        return self.inner_cv, seed

    def _calculatePerformance(
        self, feature_test: pd.DataFrame, target_test: pd.DataFrame, best_rf: RandomForestRegressor
    ):
        """
        Description
        -----------
        Function to calculate the performance metrics used to verify models

        Parameters
        ----------
        feature_test (pd.DataFrame)      pd.DataFrame of feature values (x) from the test set
        target_test (pd.DataFrame)       pd.DataFrame of targets (y) from the test set
        best_rf (object)                 RF model from the current resample

        Returns
        -------
        Series of performance metrics-
                                        1. Bias
                                        2. Standard Error of Potential
                                        3. Mean Squared Error
                                        4. Root Mean Squared Error (computed from SDEP and Bias)
                                        5. pearson R coefficient
                                        6. Spearman R coefficient
                                        7. r2 score

        """

        # Get predictions from the best model in each resample
        predictions = best_rf.predict(feature_test)

        # Calculate Errors
        true = target_test.astype(float)
        pred = predictions.astype(float)
        errors = true - pred

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        r_pearson, p_pearson = pearsonr(true, pred)

        self.logger.debug(
            "Performance metrics:"
            f"\nBias = {bias}"
            f"\nSDEP = {sdep}"
            f"\nMSE = {mse}"
            f"\nRMSE = {rmse}"
            f"\nR2 = {r2}"
            f"\nPearson R = {r_pearson}"
                           )

        return bias, sdep, mse, rmse, r2, r_pearson, p_pearson, true, pred

    def _makeModel(
        self,
        n: int,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        test_size: float,
        save_interval_models: bool,
        save_path: str,
        hyper_params: dict,
        set_seed: int=None,

    ):
        """
        Description
        -----------
        Function to carry out single resample and evaluate the performance of the predictions

        Parameters
        ----------
        n (int)                      Resample number
        features (pd.DataFrame)      Training features used to make predictions
        targets (pd.DataFrame)       Training targets used to evaluate training
        test_size (float)            Decimal of test set size (0.3 = 70/30 train/test split)
        save_interval_models (bool)  Flag to save the best rf model from each resample
        save_path (str)              Pathway to save interval models to

        Returns
        -------
        1: best parameters from the hyperparameters search
        2: Performance metrics from the best RF from the given resample
        3: Feature importances from each RF
        """

        # Setting a random seed value
        if not set_seed:
            seed = rand.randint(0, 2**31)
        else:
            seed=set_seed

        self.logger.debug(f"Random seed: {seed}")

        resample_number = n + 1

        # Doing the train/test split
        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features, targets, test_size=test_size, random_state=seed
        )

        chembl_row_indices = [
            i for i, id in enumerate(tar_te.index) if id.startswith("CHEMBL")
        ]

        # Convert DataFrames to NumPy arrays if necessary
        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        # Initialize the model and inner cv and pipeline it to prevent data leakage
        rf = Pipeline([("rf", RandomForestRegressor())])

        self.inner_cv_, _ = self._setInnerCV(
            set_seed=seed
            )

        # Setting the search type for hyper parameter optimisation
        if self.search_type == "grid":
            search = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                cv=self.inner_cv,
                scoring=self.scoring,
            )
        else:
            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=hyper_params,
                n_iter=self.n_resamples,
                cv=self.inner_cv,
                scoring=self.scoring,
                random_state=seed,
            )
        
        # Training the model
        search.fit(feat_tr, tar_tr)

        # Obtaining the best model in
        best_pipeline = search.best_estimator_
        best_rf = best_pipeline.named_steps["rf"]

        self.logger.info("Best model obtained")

        # Calculating the performance of each resample
        performance = self._calculatePerformance(
            target_test=tar_te, feature_test=feat_te, best_rf=best_rf
        )

        ChEMBL_perf = self._calculatePerformance(
            target_test=tar_te[chembl_row_indices],
            feature_test=feat_te.iloc[chembl_row_indices],
            best_rf=best_rf,
        )

        # Isolating the true and predicted values used in performance calculations
        # so analysis can be done on them
        true_vals_ls = performance[-2]
        pred_vals_ls = performance[-1]
        performance = performance[:-2]

        # Saving the model at each resample
        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}{n}.pkl")

        return (
            search.best_params_,
            performance,
            ChEMBL_perf,
            best_rf.feature_importances_,
            resample_number,
            true_vals_ls,
            pred_vals_ls,
        )

    def trainRegressor(
        self,
        search_type: str,
        scoring: str = "neg_mean_squared_error",
        n_resamples: int = 2,
        test_size: float = 0.3,
        hyper_params: dict = None,
        features: pd.DataFrame = None,
        targets: pd.DataFrame = None,
        save_interval_models: bool = False,
        save_path: str = None,
        save_final_model: bool = False,
        plot_feat_importance: bool = False,
        batch_size: int = 2,
        n_jobs: int=40,
        cv_kwargs_override: dict=None
    ):
        """
        Description
        -----------
        Function to train the RF Regressor model.

        Parameters
        ----------
        search_type (str)               Type of hyperparameter search:
                                        'grid' = grid search, exhaustive and more computationally expensive
                                        'random' = random search, non-exhaustive and less computationally expensive
        scoring (str)                   Loss function to map the hyperparameter optimisation to
        n_resamples (int)               Number of Outer Loop resamples
        test_size (float)               Decimal fort he train/test split. 0.3 = 70/30
        hyper_params (dict)             Dictionary of hyperparameters to optimise on
        features (pd.DataFrame)         Features to train the model on
        targets (pd.DataFrame)          Targets to train the model on
        save_interval_models (bool)     Flag to save best individual models from each resample
        save_path (str)                 Path to save individual models to
        save_final_model (bool)         Flag to save the final model after all resampling
        plot_feat_importance (bool)     Flag to plot the feature importance generated by RF model

        Returns
        -------
        1: Final Random Forect model in pickle format
        2: Best hyper parameters for the final model
        3: Dictionary of performance metrics
        4: Dataframe of feature importances
        """

        self.logger.info("Training random forest regressor model...")

        # Setting the training parameters
        self.search_type = search_type
        self.scoring = scoring
        self.n_resamples = n_resamples
        self.test_size = test_size

        # Dropping indexes which failed to dock
        targets = targets[targets[self.docking_column] != "False"]
        features = features.loc[targets.index]

        if cv_kwargs_override:
            self.cv_kwargs.update(cv_kwargs_override)

        def _processBatch(batch_indices: list):
            """
            Description
            -----------
            Wrapper function for processing a batch of resamples.
            Calls the _fit_model_and_evaluate() function for each resample index provided

            Parameters
            ----------
            batch_indices (list of int)     List of indices representing the current batch of
                                            resamples to process. Each index corresponds to
                                            a specific resample

            Returns
            -------
            List of tuples
                Each element is a tuple containing information from each resample:
                - Best parameters from hyperparameter search (dict)
                - Performance metrics from the best RFR (tuple)
                - Performance metrics on just the ChEMBL data from the best RFR (dict)
                - Feature importances from the best RFR (array)
                - Resample number (int)
                - True target values used to create the performance metrics (array)
                - Predicted target values used to create the performance metrics (array)
            """

            results_batch = []
            for n in batch_indices:
                result = self._makeModel(
                    n,
                    features,
                    targets,
                    test_size,
                    save_interval_models,
                    save_path,
                    hyper_params,
                )
                results_batch.append(result)
            return results_batch

        # Calculating the batch size for the number of resamples to submit
        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches = [
            range(i * batch_size, min((i + 1) * batch_size, n_resamples))
            for i in range(n_batches)
        ]

        # Multiprocessing the to process eatch batch
        results_batches = Parallel(n_jobs=n_jobs)(
            delayed(_processBatch)(batch) for batch in batches
        )

        # Flattening the results into a single list
        results = [result for batch in results_batches for result in batch]

        # Obtaining each value from result's list of lists
        (
            best_params_ls,
            self.performance_list,
            self.ChEMBL_perf_list,
            feat_importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
        ) = zip(*results)

        # Putting the best parameters into a dictionary and forcing float type onto them to
        # remove potential issues
        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()

        for key, value in best_params.items():
            if key != "rf__max_features":
                best_params[key] = int(value)

        # Calculating average performance metrics across all training resamples
        self.performance_dict = {
            "Bias": round(
                float(np.mean([perf[0] for perf in self.performance_list])), 4
            ),
            "SDEP": round(
                float(np.mean([perf[1] for perf in self.performance_list])), 4
            ),
            "MSE": round(
                float(np.mean([perf[2] for perf in self.performance_list])), 4
            ),
            "RMSE": round(
                float(np.mean([perf[3] for perf in self.performance_list])), 4
            ),
            "r2": round(float(np.mean([perf[4] for perf in self.performance_list])), 4),
            "pearson_r": round(
                float(np.mean([perf[5] for perf in self.performance_list])), 4
            ),
            "pearson_p": round(
                float(np.mean([perf[6] for perf in self.performance_list])), 4
            ),
        }

        self.ChEMBL_perf_dict = {
            "Bias": round(
                float(np.mean([perf[0] for perf in self.ChEMBL_perf_list])), 4
            ),
            "SDEP": round(
                float(np.mean([perf[1] for perf in self.ChEMBL_perf_list])), 4
            ),
            "MSE": round(
                float(np.mean([perf[2] for perf in self.ChEMBL_perf_list])), 4
            ),
            "RMSE": round(
                float(np.mean([perf[3] for perf in self.ChEMBL_perf_list])), 4
            ),
            "r2": round(float(np.mean([perf[4] for perf in self.ChEMBL_perf_list])), 4),
            "pearson_r": round(
                float(np.mean([perf[5] for perf in self.ChEMBL_perf_list])), 4
            ),
            "pearson_p": round(
                float(np.mean([perf[6] for perf in self.ChEMBL_perf_list])), 4
            ),
        }

        # Calculating average feature importances across all training resamples
        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(), "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)

        # Plotting the feature importances in a bar plot
        if plot_feat_importance:
            self.logger.info("Plotting feature importance")
            self._plotFeatureImportance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        # Removing the 'rf__' prefix on the best determined hyper parameters
        # ('rf__' needed for the pipelining of the rf model in _fit_model_and_evaluate function)
        cleaned_best_params = {
            key.split("__")[1]: value for key, value in best_params.items()
        }

        # Training final model on best hyper parameters and on whole data
        self.final_rf = RandomForestRegressor(**cleaned_best_params)
        self.final_rf.fit(features, targets.to_numpy())

        # Saving model, performance, best hyper parameters, training features and targets
        if save_final_model:
            self.logger.debug(f"Saving final model to:\n{save_path}/final_model.pkl")
            joblib.dump(self.final_rf, f"{save_path}/final_model.pkl")

            with open(f"{save_path}/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

            with open(f"{save_path}/chembl_performance_stats.json", "w") as file:
                json.dump(self.ChEMBL_perf_dict, file, indent=4)

            with open(f"{save_path}/best_params.json", "w") as file:
                json.dump(best_params, file, indent=4)

            features.to_csv(
                f"{save_path}/training_data/training_features.csv.gz",
                index_label="ID",
                compression="gzip",
            )
            targets.to_csv(
                f"{save_path}/training_data/training_targets.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        self.logger.info("Retraining complete and predictions made.")
        self.logger.info(f"Performance on total training data:\n{self.performance_dict}")
        self.logger.info(f"Performance on ChEMBL training data:\n{self.ChEMBL_perf_dict}")

        return (
            self.final_rf,
            best_params,
            self.performance_dict,
            self.ChEMBL_perf_dict,
            feat_importance_df,
            self.true_vals_ls,
            self.pred_vals_ls,
            )

    def _plotFeatureImportance(
        self,
        feat_importance_df: pd.DataFrame = None,
        top_n_feats: int = 20,
        save_data: bool = False,
        save_path: str = None,
        filename: str = None,
        dpi: int = 500,
    ):
        """
        Description
        -----------
        Function to plot feature importance

        Parameters
        ----------
        feature_importance_df (pd.DataFrame)    pd.DataFrame containing feature importances
        top_n_feats (int)                       Number of features shown
        save_data (bool)                        Flag to save plot
        save_path (str)                         Path to save plot to
        filename (str)                          Filename to save plot a
        dpi (int)                               Value for quality of saved plot

        Returns
        -------
        None
        """

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df.head(top_n_feats),
            x="Importance",
            y="Feature",
            palette="viridis",
            dodge=False,
            hue="Feature",
            legend=False,
        )

        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        if save_data:
            plt.savefig(f"{save_path}/{filename}.png", dpi=dpi)
            feat_importance_df.to_csv(f"{save_path}/feature_importance_df.csv")

        return

    def _calculateMPO(self, full_data_fpath, preds_df, preds_col_name):
        """
        Description
        -----------
        Function to get the PFI and LogP scores from the full data files and calculate the MPO values from predicted
        scores

        Parameters
        ----------
        full_data_fpath (str)       String path to the full data file
        preds_df (pd.DataFrame)     Pandas DataFrame of the predictions
        preds_col_name (str)        Column name containing the predictions

        Returns
        -------
        New pandas DataFrame object containing the MPO scores
        """
        df = pd.read_csv(
            full_data_fpath, index_col="ID", usecols=["ID", "PFI", "oe_logp"]
        )
        self.logger.debug(f"DF prior to MPO calculation:\n{df.index[:5]}")

        # Save original prediction index
        original_index = preds_df.index

        # Merge predictions with full data to align properly
        self.logger.debug(f"preds_df index sample:\n{preds_df.index[:5]}")
        self.logger.debug(f"df index sample:\n{df.index[:5]}")

        df = df.join(preds_df[[preds_col_name]], how="inner")
        self.logger.debug(f"DF after joining:\n{df.index[:5]}")


        # Round raw descriptors before MPO calculation for consistency
        for col in ["oe_logp", "PFI"]:
            if col in df.columns:
                df[col] = df[col].round(2)

        # Compute MPO on the aligned data
        df["MPO"] = [
            -score * 1 / (1 + math.exp(PFI - 8))
            for score, PFI in zip(df[preds_col_name], df["PFI"])
        ]
        
        # Reindex to match the original prediction order
        df = df.reindex(original_index)
        self.logger.debug(f"DF after MPO calculation:\n{df.index[:5]}")


        return df

    def predict(
        self,
        feats: pd.DataFrame,
        save_preds: bool,
        preds_save_path: str = None,
        preds_filename: str = None,
        final_rf: str = None,
        pred_col_name: str = "pred_Affinity(kcal/mol)",
        calc_mpo: bool = True,
        full_data_fpath: str = None,
    ):
        """
        Descripton
        ----------
        Function to take make predictions using the input RF model

        Parameters
        ----------
        feats (pd.DataFrame)        DataFrame object containing all of the features used for predictions
        save_preds (bool)           Flag to save the predictions
        preds_save_path (str)       Path to save the predictions to
        preds_filename (str)        Name to save the .csv.gz prediction dfs to
        final_rf (str)              Path to the RF pickle file used to make predictions
        pred_col_name (str)         Name of the column in filename to save predictions to

        Returns
        -------
        pd.DataFrame object containing all of the predictions
        """

        if isinstance(final_rf, (str, Path)):
            rf_model = joblib.load(final_rf)
        elif final_rf is not None:
            rf_model = final_rf
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame(
            data={pred_col_name: rf_model.predict(feats)},
            index=feats.index
        )
        self.logger.debug(f"1 Predictions head:\n{preds_df.head()}")

        preds_df.index = feats.index
        self.logger.debug(f"2 Predictions head:\n{preds_df.head()}")


        all_tree_preds = np.stack(
            [tree.predict(feats) for tree in rf_model.estimators_]
        )

        if calc_mpo:
            preds_df = self._calculateMPO(
                full_data_fpath, preds_df=preds_df, preds_col_name=pred_col_name
            )

        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)
        self.logger.debug(f"3 Predictions head:\n{preds_df.head()}")
    
     # Round numerical prediction outputs to 2 decimal places
        round_cols = [pred_col_name, "MPO", "Uncertainty"]
        for col in round_cols:
            if col in preds_df.columns:
                preds_df[col] = preds_df[col].round(2)

        if save_preds:
            preds_df.to_csv(
                f"{preds_save_path}/{preds_filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        return preds_df
