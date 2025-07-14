import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import json
from glob import glob
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import subprocess
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_squared_error, r2_score
import math
from PIL import Image as PILImage
from io import BytesIO
from scipy.spatial import ConvexHull
import random as rand
import time
import logging
import os
from collections import defaultdict
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
import colorcet as cc
import json

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
RESULTS_DIR = PROJ_DIR / 'results'
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}


# Misc
sys.path.insert(0, str(PROJ_DIR) + "/scripts/misc/")
from misc_functions import (
    molid2BatchNo,
    countIterations,
    countConformations,
    getMolsBetweenIters,
    molidList2Smiles,
    getDesc4Molid,
    fig2Img,
    getTop,
    wait4Jobs
)

# Docking
sys.path.insert(0, str(PROJ_DIR) + "/scripts/docking/")
from docking_fns import (
    wait4Docking,
    RunGNINA,
    getUndocked
)

# Dataset
sys.path.insert(0, str(PROJ_DIR) + "/scripts/dataset/")
from dataset_fns import DatasetAccessor



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

def calcStats(
    df: pd.DataFrame,
    true_col: str = "Affinity(kcal/mol)",
    pred_col: str = "pred_Affinity(kcal/mol)",
):

    # Extract True and Predicted Values from DF
    true = df[true_col].values
    pred = df[pred_col].values

    # Pearsons r^2
    pearson_r, _ = pearsonr(true, pred)
    pearson_r2 = pearson_r**2

    # Coefficient of Determination (R^2)
    cod = r2_score(true, pred)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(true, pred))

    # Bias (Systematic Error / Mean Error)
    bias = np.mean(true - pred)

    # SDEP (Standard Deviation of the Error of Prediction / Random Error)
    sdep = np.sqrt(np.mean((true - pred - bias) ** 2))

    return (true, pred, pearson_r2, cod, rmse, bias, sdep)


class Analysis:
    def __init__(
        self,
        results_dir = RESULTS_DIR,
        rdkit_or_mordred: str = "rdkit",
        held_out_stat_json: str = "held_out_stats.json",
        docking_column: str = "Affinity(kcal/mol)",
        log_level: str="DEBUG"
    ):
        """
        Description
        -----------
        Class to carry out the Recreating_DMTA workflow analysis

        Parameters
        ----------
        rdkit_or_mordred (str)      Value to set the working results directory, set as 'rdkit' or 'mordred'
        held_out_stats_json (str)   Name of the json file containing the performance stats on the held out
                                    test set
        docking_column (str)        Name of column which the docking scores are saved under

        Returns
        -------
        Initialised analysis class
        """
        self.plot_dir = results_dir / "plots"

        # Creating link between it0 of this run and the initial model
    
        if not self.plot_dir.exists():
            self.plot_dir.mkdir()
            print(f"Made plot dir:\n{self.plot_dir}")

        # Setting up logging
        log_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
        self.logger = logging.getLogger(f"SimDMTA_Analysis")
        self.logger.setLevel(log_level)

        log_file = RESULTS_DIR / "SimDMTA_analysis.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(funcName)s | Line %(lineno)d | \n%(message)s\n'))
        file_handler.setLevel(log_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)

        self.logger.propagate = False

        self.logger.info(f"Logging to: {log_file}")

        self.rdkit_or_mordred = rdkit_or_mordred.lower()

        self.results_dir = results_dir

        self.held_out_stat_json = held_out_stat_json

        self.docking_column = docking_column

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        self.linestyles = {"_10_": "--", "_50_": "-"}
        self.method_colour_map = {
                "_mp": colours[0],
                "_mu": colours[1],
                "_r": colours[2],
                "_rmp": colours[3],
                "_rmpo": colours[4],
                "_mpo": colours[5],
                "_rmu": colours[6],

                "_0025": colours[7],
                "_005": colours[8],
                "_01": colours[3],
                "_025": colours[9],
                "_05": colours[10],

                "_2:8": colours[11],
                "_5:5": colours[12],
                "_8:2": colours[13],
            }
        
        self.logger.debug(f"Colour map: {self.method_colour_map}")

    def _getStats(
        self,
        experiment_dirs: list = [],
        perf_stats_json: str = "performance_stats.json",
    ):
        """
        Description
        -----------
        Function to get the performance statistics for all experiments on both the internal and held out tests

        Parameters
        ----------
        experiment_dirs (list)      List of experiment names e.g., [20240910_10_r, 20241012_10_r]

        Returns
        -------
        Dictionary containing all available iteration statistics on internal and held out tests for each given expetiment
        """

        all_stats = {}

        # Looping through all provided experiments
        for exp in experiment_dirs:
            with open(Path(exp) / "run_params.json", "r") as f:
                param_dict = json.load(f)
            
            step = int(param_dict["batch_size"])

            # Initialising empty lists
            rmse = []
            r2 = []
            bias = []
            sdep = []
            pearson = []

            no_mols_ls = []

            # Defining the working directory
            working_dir = self.results_dir / exp

            # For each iteration obtain and save the statistics data
            # If this doesnt work change back to (0, cnt_n_iters())
            for n in range(0, countIterations(working_dir)):
                no_mols_ls.append(n * step)

                stats_path = f"{working_dir}/it{n}/{perf_stats_json}"

                try:
                    with open(stats_path, "r") as perf_stats:
                        data = json.load(perf_stats)

                    rmse.append(round(float(data.get("RMSE", 0)), 3))
                    r2.append(round(float(data.get('r2', 0)), 3))
                    bias.append(round(float(data.get("Bias", 0)), 3))
                    sdep.append(round(float(data.get("SDEP", 0)), 3))
                    pearson.append(round(float(data.get('pearson_r', 0)), 3))


                except Exception as e:
                    self.logger.warning(e)

            # Format the statistics data
            all_stats[exp] = {
                "n_mols": no_mols_ls,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
                "sdep": sdep,
                "pearson_r": pearson
            }
        
        self.logger.debug(f"Calculated statistics: {all_stats}")

        return all_stats

    def plotModelPerformance(
        self,
        experiments: list,
        save_plot: bool = True,
        method_legend_map: dict = None,
        plot_fname: str = "Perf_Plot",
        plot_int: bool = False,
        plot_ho: bool = False,
        plot_chembl_int: bool = False,
        set_ylims: bool = True,
        r2_ylim: tuple = (-1, 1),
        bias_ylim: tuple = (-0.5, 0.5),
        rmse_ylim: tuple = (0, 1),
        sdep_ylim: tuple = (0, 1),
        r_type: str = 'r2',
        xticks: int=None,
        yticks: int=None,
        tick_fontsize: int=10,
        label_fontsize:int=12,
        title_fontsize:int=14,
        legend_fontsize:int=10,
        font_family: str= "DejaVu Sans",
        custom_xticks: list=None,
        linewidth:int=2,

    ):
        
        self.logger.info("Plotting model predictive performance plots")

        plt.rcParams.update({
            "font.family": font_family
        })

        # Load performance stats for the selected datasets
        all_int_stats = (
            self._getStats(
                experiment_dirs=experiments, perf_stats_json="performance_stats.json",
            )
            if plot_int
            else None
        )

        all_ho_stats = (
            self._getStats(
                experiment_dirs=experiments,
                perf_stats_json=self.held_out_stat_json,
            )
            if plot_ho
            else None
        )

        all_chembl_stats = (
            self._getStats(
                experiment_dirs=experiments,
                perf_stats_json="chembl_performance_stats.json",
            )
            if plot_chembl_int
            else None
        )

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        # Determine the maximum length of metrics from any of the datasets to pad missing data
        max_length = 0
        for stats in [all_int_stats, all_ho_stats, all_chembl_stats]:
            if stats:
                max_length = max(
                    max_length, max(len(s["rmse"]) for s in stats.values())
                )
        
        colour_ls = []
        for exp in experiments:
            with open(Path(exp) / "run_params.json", "r") as f:
                param_dict = json.load(f)
            step = int(param_dict["batch_size"]) 

            name = exp.split("_")[-1]
            exp_name = f"_{name}" 
            method = next((m for m in self.method_colour_map.keys() if exp.endswith(m)), None)
            colour = self.method_colour_map.get(method, "black")
            colour_ls.append(colour)
            linestyle = self.linestyles.get("_50_" if "_50_" in exp else "_10_", "--")

            self.logger.debug(f"Experiment: {exp}, Color: {colour}, Line Style: {linestyle}")

            def plot_metric(ax, stats, metric, linestyle, linewidth, color, label=None):
                padded_data = np.pad(
                    stats[exp][metric],
                    (0, max_length - len(stats[exp][metric])),
                    constant_values=np.nan,
                )
                sns.lineplot(
                    x=list(range(0, max_length * step, step)),
                    y=padded_data,
                    legend=False,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    label=label,
                    ax=ax,
                )

            if plot_int:
                plot_metric(
                    ax[0, 0],
                    all_int_stats,
                    "rmse",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_int_stats,
                    r_type,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_int_stats,
                    "bias",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_int_stats,
                    "sdep",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )

            if plot_ho:
                plot_metric(
                    ax[0, 0],
                    all_ho_stats,
                    "rmse",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_ho_stats,
                    r_type,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_ho_stats,
                    "bias",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_ho_stats,
                    "sdep",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )

            if plot_chembl_int:
                plot_metric(
                    ax[0, 0],
                    all_chembl_stats,
                    "rmse",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_chembl_stats,
                    r_type,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_chembl_stats,
                    "bias",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_chembl_stats,
                    "sdep",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=colour,
                    label=exp_name,
                )

            #ax[0, 0].set_title("RMSE", fontsize=title_fontsize)
            ax[0, 0].set_ylabel("RMSE", fontsize=label_fontsize, labelpad=10)

            #ax[1, 0].set_title(r_type, fontsize=title_fontsize)
            if r_type == "pearson_r":
                ax[1, 0].set_ylabel("Pearson R", fontsize=label_fontsize, labelpad=10)
            else:
                ax[1, 0].set_ylabel("R\u00b2", fontsize=label_fontsize, labelpad=10)


            #ax[1, 1].set_title("Bias", fontsize=title_fontsize)
            ax[1, 1].set_ylabel("Bias", fontsize=label_fontsize, labelpad=10)

            #ax[0, 1].set_title("SDEP", fontsize=title_fontsize)
            ax[0, 1].set_ylabel("SDEP", fontsize=label_fontsize, labelpad=10)

            if set_ylims:
                ax[0, 0].set_ylim(rmse_ylim[0], rmse_ylim[1])
                ax[1, 0].set_ylim(r2_ylim[0], r2_ylim[1])
                ax[1, 1].set_ylim(bias_ylim[0], bias_ylim[1])
                ax[0, 1].set_ylim(sdep_ylim[0], sdep_ylim[1])

            for a in ax.flat:
                # For top row: remove only x-axis tick labels and x-axis label
                if a in [ax[0, 0], ax[0, 1]]:
                    a.set_xlabel("")
                    a.tick_params(axis="x", labelbottom=False)  # Hide x-axis tick labels only
                else:
                    a.set_xlabel("Molecule Count", fontsize=label_fontsize)

                a.tick_params(axis="both", labelsize=tick_fontsize)

                if custom_xticks is not None:
                    a.set_xticks(custom_xticks)
                    a.tick_params(axis='x', labelrotation=45)  # rotate x-axis tick labels
                elif xticks is not None:
                    a.xaxis.set_major_locator(plt.MaxNLocator(xticks))

                if yticks is not None:
                    a.yaxis.set_major_locator(plt.MaxNLocator(yticks))
                    

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="-"),
            plt.Line2D([0], [0], color="black", linestyle="--"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        leg1 = fig.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
            prop={"size": legend_fontsize}
        )

        # Track which suffixes are present in the current experiments
        used_suffixes = set()
        for exp in experiments:
            matched_suffix = next((s for s in self.method_colour_map if exp.endswith(s)), None)
            if matched_suffix:
                used_suffixes.add(matched_suffix)

        # Build handles and labels in defined order
        handles = []
        labels = []
        if method_legend_map:
            for suffix, label in method_legend_map.items():
                if suffix in used_suffixes:
                    colour = self.method_colour_map.get(suffix)
                    if colour:
                        handles.append(Line2D([0], [0], color=colour, lw=2))
                        labels.append(label)
                    else:
                        self.logger.warning(f"Warning: No color found for method suffix {suffix}")

        leg2 = fig.legend(
            handles,
            [label.lstrip('_') for label in labels],
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
            prop={"size": legend_fontsize}
        )

        fig.add_artist(leg1)
        fig.add_artist(leg2)

        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_plot:
            path = self.plot_dir / f"{plot_fname}"
            self.logger.info(path)
            plt.savefig(path, dpi=600, bbox_inches="tight")

        plt.show()

    def plotPCA(
        self,
        train: str,
        prediction: str,
        source_ls: list,
        validation: str = None,
        n_components: int = 5,
        loadings_filename: str = "pca_loadings",
        pca_df_filename: str = "pca_components",
        kdep_sample_size: float = 0.33,
        contamination: float = 0.00001,
        plot_fname: str = "plotPCA",
        save_plot: bool = True,
        save_extra_data: bool = False,
        plot_area: bool = False,
        plot_scatter: bool = True,
        random_seed: int = None,
        plot_loadings: bool = False,
        plot_title: str = 'PCA Plot',
        remove_outliers: bool = True,
        kdep_sample_ls: list = ['PyMolGen'],
        axis_fontsize: int = 20,
        tick_fontsize: int = 18,
        label_fontsize: int = 20,
        legend_fontsize: int = 20,
        kde_tick_dicts: list = None
    ):
        self.logger.info("Plotting PCA...")

        if random_seed is None:
            random_seed = rand.randint(0, 2**31)

        # Load and tag training data
        if isinstance(train, (str, Path)):
            train_df = pd.read_csv(train, index_col="ID")
        else:
            train_df = train.copy()
        train_df["Source"] = source_ls[0]

        # Load prediction data
        if isinstance(prediction, str):
            pred_files = glob(prediction)
            prediction_df = pd.DataFrame()
            for file in pred_files:
                df = pd.read_csv(file, index_col="ID")
                prediction_df = pd.concat([prediction_df, df], axis=0)
        elif isinstance(prediction, Path):
            prediction_df = pd.read_csv(prediction, index_col="ID")
        else:
            prediction_df = prediction.copy()
        prediction_df["Source"] = source_ls[-1]

        # Load validation data if provided
        if validation is not None:
            if isinstance(validation, (str, Path)):
                validation_df = pd.read_csv(validation, index_col="ID")
            else:
                validation_df = validation.copy()
            validation_df["Source"] = source_ls[1]
        else:
            validation_df = pd.DataFrame()

        # Identify common descriptor columns (exclude Source)
        common_cols = set(train_df.columns) & set(prediction_df.columns)
        if not validation_df.empty:
            common_cols &= set(validation_df.columns)
        common_cols -= {"Source"}  # exclude Source from features

        # Reduce each DataFrame to common columns + Source
        common_cols = list(common_cols)
        train_df = train_df[common_cols + ["Source"]]
        prediction_df = prediction_df[common_cols + ["Source"]]
        if not validation_df.empty:
            validation_df = validation_df[common_cols + ["Source"]]

        # Combine all data (sorted by size to control plotting layer order)
        df_dict = {
            "train": train_df,
            "validation": validation_df,
            "prediction": prediction_df
        }
        sorted_dfs = sorted(df_dict.items(), key=lambda x: len(x[1]), reverse=True)
        combined_df = pd.concat([df for _, df in sorted_dfs], axis=0)
        combined_df = combined_df.dropna()

        self.logger.debug(f"Combined data prior to generation:\n{combined_df}")

        # Perform scaling on feature columns only
        used_cols = [col for col in combined_df.columns if col != "Source"]
        scaler = StandardScaler()
        scaled_combined_df = pd.DataFrame(
            scaler.fit_transform(combined_df[used_cols]),
            columns=used_cols,
            index=combined_df.index
        )
        scaled_combined_df["Source"] = combined_df["Source"]

        self.logger.debug(f"Scaled and combined data prior to generation:\n{scaled_combined_df}")

        # Doing the PCA on the data
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_combined_df[used_cols])
        explained_variance = pca.explained_variance_ratio_ * 100

        # Isolating the loadings for each principal components and labelling the associated features
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=used_cols
        )
        
        abs_loadings_df = loadings_df.abs()
        abs_loadings_df.rename_axis("Features", inplace=True)
        
        loadings_df.rename_axis("Features", inplace=True)

        if save_extra_data:
            loadings_df.to_csv(
                self.plot_dir / f"{loadings_filename}.csv", index_label="Features"
            )
            abs_loadings_df.to_csv(
                self.plot_dir / f"{loadings_filename}_abs.csv", index_label="Features"
            )
        

        # Creating a DataFrame for the principal component results. Saves to .csv
        pca_df = pd.DataFrame(
            principal_components,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=combined_df.index,
        )
        pca_df["Source"] = combined_df["Source"].values
        dark_colours = sns.color_palette('dark')
        source_color_map = {src: dark_colours[i] for i, src in enumerate(source_ls)}

        if save_extra_data:
            pca_df.to_csv(self.plot_dir / f"{pca_df_filename}.csv.gz", index_label="ID", compression='gzip')

        # Removing outlying molecules from the PCA data
        def remove_outliers(df, columns, n_neighbors=20, contamination=contamination):
            """
            Description
            -----------
            Function to remove outlying molecules from a DataFrame

            Parameters
            ----------
            df (pd.DataFrame)       DataFrame from which you wish to remove the outlying molecules
            columns (list)          Columns you want to consider when defining outliers
            contamination (float)   Fraction of outlying points you wish to remove from the dataset

            Returns
            -------
            New DataFrame with outlying molecules removed

            """
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )
            outlier_labels = lof.fit_predict(df[columns])
            return df[outlier_labels == 1]

        if remove_outliers:
            pca_df = remove_outliers(pca_df, [f"PC{i+1}" for i in range(n_components)])

        # Initialise PCA subplots
        fig, axs = plt.subplots(
            nrows=n_components, ncols=n_components, figsize=(20, 20)
        )

        # Filling in the subplots
        for i in range(n_components):
            for j in range(n_components):

                # If not on the diagonal, make a scatter plot for the PCA overlap
                if i != j:
                    
                    if plot_scatter:
                        sns.scatterplot(
                                x=f"PC{j+1}",
                                y=f"PC{i+1}",
                                hue="Source",
                                data=pca_df,
                                ax=axs[i, j], 
                                legend=False,
                                edgecolor="none",
                                palette=source_color_map,
                                alpha=0.5
                            )
                    
                    dark_colours = sns.color_palette('dark')
                    sorted_sources = [s for s in source_ls if s in pca_df["Source"].unique()]

                    for idx, source in enumerate(sorted_sources):
                        source_data = pca_df[pca_df['Source'] == source]
                        area_colour = source_color_map[source]

                                            
                        if plot_area:
                            # Calculate convex hull of points
                            points = source_data[[f"PC{j+1}", f"PC{i+1}"]].values
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]

                            # Close the polygon by appending the first point
                            hull_points = np.vstack((hull_points, hull_points[0]))
                            
                            # Plot the area
                            axs[i, j].fill(
                                hull_points[:, 0],
                                hull_points[:, 1],
                                alpha=0.2,  # Transparency for the filled area
                                color=area_colour,
                                edgecolor = area_colour,
                                linewidth=2,
                                label=f"{source} area"
                            )


                    # --- Common tick settings for all subplots
                    axs[i, j].tick_params(axis='both', labelsize=tick_fontsize, pad=6)

                    if i == j:
                        # Diagonal (KDE)
                        axs[i, i].set_xlabel(f"PC{i+1}", fontsize=axis_fontsize, labelpad=10)
                        axs[i, i].set_ylabel("Density", fontsize=label_fontsize, labelpad=10)

                        if kde_tick_dicts and i < len(kde_tick_dicts):
                            tick_info = kde_tick_dicts[i]
                            if "xticks" in tick_info:
                                axs[i, i].set_xticks(tick_info["xticks"])
                            if "yticks" in tick_info:
                                axs[i, i].set_yticks(tick_info["yticks"])
                    else:
                        # Off-diagonal (scatter or area)
                        if i == n_components - 1:
                            axs[i, j].set_xlabel(f"PC{j+1} ({explained_variance[j]:.2f}% Var)", fontsize=axis_fontsize, labelpad=10)
                        else:
                            axs[i, j].set_xlabel("")
                            axs[i, j].set_xticklabels([])

                        if j == 0:
                            axs[i, j].set_ylabel(f"PC{i+1} ({explained_variance[i]:.2f}% Var)", fontsize=axis_fontsize, labelpad=10)
                        else:
                            axs[i, j].set_ylabel("")
                            axs[i, j].set_yticklabels([])

                # If on the diagonal, make the Kernel Density Estimate Plots for each Principal Component
                else:
                    # Because this is slow, you can take a sample of the principal component data rather than using the full data
                    src1_data = pca_df[pca_df["Source"] == source_ls[0]]

                    if source_ls[0] in kdep_sample_ls:
                        subset_src1_data = src1_data.sample(
                            n=int(len(src1_data) * kdep_sample_size),
                            random_state=random_seed
                        )
                    else:
                        subset_src1_data = src1_data

                    src2_data = pca_df[pca_df["Source"] == source_ls[1]]
                    if source_ls[1] in kdep_sample_ls:
                        subset_src2_data = src2_data.sample(
                            n=int(len(src2_data) * kdep_sample_size),
                            random_state=random_seed

                        )
                    else:
                        subset_src2_data=src2_data

                    try:
                        src3_data = pca_df[pca_df["Source"] == source_ls[2]]

                        if source_ls[2] in kdep_sample_ls:
                            subset_src3_data = src3_data.sample(
                                n=int(len(src3_data) * kdep_sample_size),
                                random_state=random_seed

                            )
                        else: 
                            subset_src3_data = src3_data

                    except IndexError:
                        subset_src3_data = pd.DataFrame()        

                    sampled_pca_df = pd.concat(
                        [subset_src1_data, subset_src2_data, subset_src3_data], axis=0
                    )

                    # Making the Kernel Density Estimate Plot
                    sns.kdeplot(
                        x=f"PC{i+1}",
                        hue="Source",
                        data=sampled_pca_df,
                        common_norm=False,
                        fill=True,
                        ax=axs[i, i],
                        legend=False,
                        palette=source_color_map,
                    )

                    if kde_tick_dicts and i < len(kde_tick_dicts):
                        tick_info = kde_tick_dicts[i]
                        if "xticks" in tick_info:
                            axs[i, i].set_xticks(tick_info["xticks"])
                        if "yticks" in tick_info:
                            axs[i, i].set_yticks(tick_info["yticks"])
                        
                        axs[i, i].tick_params(axis='both', labelsize=tick_fontsize)
                            

                axs[i, i].set_xlabel("")
                axs[i, i].set_ylabel("Density", fontsize=label_fontsize, labelpad=10)
                axs[i, i].tick_params(axis='both', labelsize=tick_fontsize)

                # Adjusting labels and titles, including the variance for each principal component
                if i == n_components - 1:
                    axs[i, j].set_xlabel(
                        f"PC{j+1} ({explained_variance[j]:.2f}% Var)", fontsize=axis_fontsize, labelpad=10
                    )
                if j == 0:
                    axs[i, j].set_ylabel(
                        f"PC{i+1} ({explained_variance[i]:.2f}% Var)", fontsize=axis_fontsize, labelpad=10
                    )

        # Define handles and labels for the legend
        fig = plt.gcf()  # Get current figure
        ax = axs[0, 0]  # Use first subplot as reference

        # Create custom legend handles
        custom_handles = []
        custom_labels = []

        for idx, source in enumerate(pca_df['Source'].unique()):
            # Scatter point handle
            scatter_handle = plt.Line2D([], [], 
                color=source_color_map[source],
                marker='o',
                linestyle='None',
                markersize=10,          # Bigger marker
                markeredgecolor='black',
                markeredgewidth=0.8,
                label=source
            )

            
            # Area patch handle
            area_handle = plt.Rectangle((0, 0), 1.5, 1.5,
                color=dark_colours[idx],
                alpha=0.2,
                label=f"{source} area"
            )

            
            custom_handles.extend([scatter_handle, area_handle])
            custom_labels.extend([source, f"{source} area"])

        # Place legend
        fig.legend(
            custom_handles, 
            custom_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.00),  # Pushes legend above plot
            ncol=3,
            fontsize=legend_fontsize,
            frameon=False
        )


        #fig.suptitle(plot_title, fontsize=16, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,
            right=0.9,
            top=0.95,
            bottom=0.3,
            wspace=0.6,
            hspace=0.6
        )

        if save_plot:
            plt.savefig(
                self.plot_dir /  f"{plot_fname}.png",
                dpi=600,
                bbox_inches='tight'
            )

        if plot_loadings:
            loadings_df['Max'] = loadings_df.max(axis=1)
            if isinstance(loadings_df, pd.DataFrame):
                loadings_df = loadings_df[loadings_df['Max'] > 0.3]
                loadings_df.drop(columns=['Max'])
            

            fig, ax = plt.subplots(n_components, 1, figsize=(25,25), sharex=True)

            for n in range(1, n_components + 1):
                sns.barplot(x=abs_loadings_df.index, y=abs_loadings_df[f'PC{n}'], ax=ax[n-1])
                ax[n-1].set_ylabel(f"PC{n} Loadings", labelpad=10)

            ax[n-1].set_xticklabels(range(1, len(abs_loadings_df) + 1), rotation=90)

            # Create a legend mapping numbers to feature names
            legend_labels = [f"{i+1}: {feature}" for i, feature in enumerate(loadings_df.index)]
            fig.legend(legend_labels, loc="center right", title="Feature Legend", fontsize=legend_fontsize)

            # Add a shared xlabel
            fig.supxlabel("Features (Mapped to Index Numbers)")
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, bottom=0.2, right=0.85, top=0.95, wspace=0.4, hspace=0.4)
              # Leave space for the legend
            plt.savefig(
                self.plot_dir /  f"{plot_fname}_loadings.png",
                dpi=600,
                bbox_inches='tight'
                )


        return fig

    def conformerAnalysis(
        self,
        docking_dir: str = PROJ_DIR / "docking" / "PyMolGen",
        sample_size: int = 15,
        conf_gen_plot: bool = True,
        score_convergence: bool = True,
        save_path: str=""
    ):
        """
        Description
        -----------
        Function to look into the conformer generation process. Plots the number of conformers generated for a sample of molecules
        along with the number of docked conformers. Can also plot the highest docking score by iteration to see how far through the
        conformer search are we finding the best docking scores.

        Parameters
        ---------
        docking_dir (str)           Directory to find all of the docking .tar.gz files in
        sample_size (int)           Number of molecules to consider at any given time
        conf_gen_plot (bool)        Flag to plot the number of conformers made (and docked) for each molecule
        score_convergence (bool)    Flag to plot the convergence of highest docking scores by iteration

        Returns
        -------
        None
        """

        self.logger.info("Running conformer analysis...")

        # Initiailising empty lists to be used
        n_confs_ls = []
        n_docked_ls = []
        molid_ls = []
        scores_ls = []
        pose_ls = []

        # Obtaining all of the molid files available
        tar_files = glob(docking_dir + "PMG*.tar.gz")

        # Taking a random sample of molecules from all available
        for file in random.sample(tar_files, sample_size):
            file = Path(file)

            # Obtaining just the Molecule ID (.stem removes the .gz suffix)
            molid = file.stem[:-4]

            # Make temporary directory to investigate data in
            output_dir = PROJ_DIR / "docking" / "PyMolGen" / f"extracted_{molid}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # tar command to unzip and untar the molecule docking dataa
            command = ["tar", "-xzf", str(file), "-C", str(output_dir)]
            try:
                subprocess.run(command, check=True)

                # Unzip the .csv.gz file
                try:
                    # Trying to unzip the all_scores file, if fails continues onto next molecule ID
                    gz_file = output_dir / molid / f"{molid}_all_scores.csv.gz"
                    docked_confs_df = pd.read_csv(gz_file)

                    # Updating lists with necessary data
                    scores_ls.append(docked_confs_df[self.docking_column].tolist())
                    n_docked_confs = len(docked_confs_df)
                    n_docked_ls.append(n_docked_confs / 9)
                    pose_ls.append(docked_confs_df.index)

                    # Counting the number of conformations in the all_confs .sdf file
                    n_total_confs = countConformations(
                        f"{output_dir}/{molid}/all_confs_{molid}_pH74.sdf"
                    )
                    n_confs_ls.append(n_total_confs)
                except:
                    continue

                # Remove the extracted directory
                rm_command = ["rm", "-r", str(output_dir)]
                subprocess.run(rm_command, check=True)

                # Adding the molecule ID to the list if successful
                molid_ls.append(molid)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to extract {file}. Error: {e}")

        if conf_gen_plot:

            # Creating a pd.DataFrame with all of the necessary data to make the
            # number of conformers per molecule ID plot

            conf_df = pd.DataFrame(
                {
                    "n_confs_made": n_confs_ls,
                    "molids": molid_ls,
                    "n_confs_docked": n_docked_ls,
                }
            )

            conf_df.index = molid_ls

            # Making the scatter plots
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_made")
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_docked")

            # Formatting the scatter plots
            plt.title("Conformer Generation Analysis")
            plt.xticks(rotation=90)
            plt.ylabel("Number of conformers made")
            plt.xlabel("Molecule ID")
            plt.tight_layout()
            plt.savefig(f"{save_path}/conf_gen_plot.png")
            plt.show()

        if score_convergence:

            # Initialising an empty list for all normalised scores
            all_norm_score_lists = []

            for ds_ls in scores_ls:

                # Finding the best scores after each iteration
                best_score = 0
                best_score_ls = []
                for score in ds_ls:
                    if score <= best_score:
                        best_score = score
                    best_score_ls.append(best_score)

                # Normalising the scores between 0 and 1
                min_score = min(best_score_ls)
                max_score = max(best_score_ls)
                if max_score == min_score:
                    normalised_scores = [0.5] * len(best_score_ls)
                else:
                    normalised_scores = [
                        (score - min_score) / (max_score - min_score)
                        for score in best_score_ls
                    ]

                # Updating the normalised scores list
                all_norm_score_lists.append(normalised_scores)

            # Plot the best score lists
            plt.figure()
            for best_score_ls, molid in zip(all_norm_score_lists, molid_ls):
                plt.plot(best_score_ls, label=molid, alpha=0.5)

            # Formatting the plots
            plt.xlabel("Pose Number")
            plt.ylabel("Best Score")
            plt.title("Best Scores Over Time")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)

            plt.tight_layout()
            plt.savefig(
                self.plot_dir / "conf_conv_plot.png"
            )
            plt.show()

        return

    def predictionDevelopment(
        self,
        experiment: str,
        iter_ls: list,
        n_plots: int=16,
        prediction_fpath: str = "held_out_preds.csv",
        true_path: Path = PROJ_DIR / "datasets" / "held_out_data" / "PMG_held_out_targ_trimmed.csv",
        it0_dir = None,
        dot_size: int = 3,
        save_plot: bool=True,
        plot_filename: str = "preds_dev_plot.png",
        tl_box_position: tuple = (0.45, 0.92),
        br_box_position: tuple =(0.95, -0.1),
        underlay_it0: bool=False,
        regression_line_colour: str = 'green',
        x_equals_y_line_colour: str = 'red',
        it0_dot_colour: str='purple',
        it_dot_colour: str='teal',
        title_fontsize: int=18,
        tick_fontsize: int=18,
        label_fontsize: int=18,
        legend_fontsize: int=18,
        metric_fontsize=15,
        x_ticks: int=None,
        y_ticks: int=None,
        figsize:tuple=(14,14),
    ):
        """
        Description
        -----------
        Function to look at the true vs predicted values over the iterations for a given test set.
        It will take an even distribution of iteration data from n number of iterations and plot them in a grid.

        Parameters
        ----------
        experiment (str)        Name of the experiment (e.g., 20240910_10_r)
        iter_ls (list)          Iterations to plot recommended for 10 and 50 molecule selections are as follows:
                                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        n_plots (int)           Number of plots to make/iterations to consider. Needs to be a square number (e.g., 4, 9, 16, etc.)
        prediction_fpath (str)  Path to the predicted values for each iteration, considering the pathing is the same.
                                (Only need to specify the dirs after the iteration dir e.g.,
                                DO -> '/held_out_test/preds.csv
                                NOT -> '/path/to/results/it{n}/held_out_test/preds.csv)
        true_path (str)         Pathway to true data .csv file
        dot_size (int)          Size of the dots on the scatter plot
        plot_filename (str)     Name to save the plot under

        Returns
        -------
        None
        """

        ""

        with open(Path(experiment) / "run_params.json", "r") as f:
            param_dict = json.load(f)

        sel_method = str(param_dict["selection_method"])
        step = int(param_dict["batch_size"])
        # Defining the results directory
        working_dir = self.results_dir / experiment

        n_y_plots = int(np.sqrt(n_plots))
        n_x_plots = n_y_plots

        # Reading in the true values
        true_scores = pd.read_csv(true_path, index_col="ID")[self.docking_column]
    
        its_to_plot = iter_ls

        # Initialising the subplots
        fig, ax = plt.subplots(nrows=n_x_plots, ncols=n_y_plots, figsize=figsize)

        full_it0_preds_path = it0_dir / prediction_fpath

        it0_preds_df = pd.read_csv(full_it0_preds_path, index_col='ID')
        it0_df = pd.DataFrame()
        it0_df[self.docking_column] = true_scores
        it0_df[f'pred_{self.docking_column}'] = it0_preds_df[f"pred_{self.docking_column}"].tolist()

        # Saving the prediciton dataframes
        df_list = []
        for it in its_to_plot:
            it_dir = working_dir / f"it{it}"
            preds = it_dir / prediction_fpath
            pred_df = pd.read_csv(preds, index_col="ID")
            pred_df[self.docking_column] = true_scores
            df_list.append(pred_df)

        # Plotting the results
        for i, (df, iter) in enumerate(zip(df_list, its_to_plot)):
            row = i // n_x_plots
            col = i % n_y_plots

            if underlay_it0:
                sns.scatterplot(
                    data=it0_df,
                    x=self.docking_column,
                    y=f'pred_{self.docking_column}',
                    ax=ax[row,col],
                    s=dot_size,
                    color=it0_dot_colour
                )

            sns.scatterplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                s=dot_size,
                color=it_dot_colour
            )

            # Add line of best fit using regplot
            sns.regplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                scatter=False,
                line_kws={"linestyle": "-", "color": regression_line_colour},
            )

            # Calculate the slope of the line of best fit (regression line)
            slope, intercept = np.polyfit(
                df[self.docking_column], df[f"pred_{self.docking_column}"], 1
            )

            # Plot y=x line for reference
            min_val = min(
                df[self.docking_column].min(), df[f"pred_{self.docking_column}"].min()
            )
            max_val = max(
                df[self.docking_column].max(), df[f"pred_{self.docking_column}"].max()
            )
            ax[row, col].plot(
                [min_val, max_val], [min_val, max_val], color=x_equals_y_line_colour, linestyle="-"
            )

            # Calculate stats
            true, pred, pearson_r2, cod, rmse, bias, sdep = calcStats(df)

            avg_pred = np.mean(pred)

            ax[row, col].set_title(f"{iter * step} mols", fontsize=label_fontsize)

            br_textstr = (
                f"$R^2_{{cod}}$: {cod:.2f}\n"
                f"$R_{{pear}}$: {pearson_r2:.2f}\n"
            )

            tl_textstr = (
                f"RMSE: {rmse:.2f}\n"
                f"Grad: {round(slope, 2)}"
            )

            ax[row, col].text(
                br_box_position[0],
                br_box_position[1],
                br_textstr,
                transform=ax[row, col].transAxes,
                fontsize=metric_fontsize,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

            ax[row, col].text(
                tl_box_position[0],
                tl_box_position[1],
                tl_textstr,
                transform=ax[row, col].transAxes,
                fontsize=metric_fontsize,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, col].set_aspect('equal', adjustable='box')

            # Remove axis labels completely from subplots (but keep ticks)
            ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel("")

            # Set tick label font size for edge plots only
            if row != n_x_plots - 1:  # Not bottom row → hide x tick labels
                ax[row, col].tick_params(labelbottom=False)
            else:
                ax[row, col].tick_params(axis='x', labelsize=tick_fontsize)

            if col != 0:  # Not left column → hide y tick labels
                ax[row, col].tick_params(labelleft=False)
            else:
                ax[row, col].tick_params(axis='y', labelsize=tick_fontsize)

            # Apply custom tick positions (for all subplots)
            if x_ticks is not None:
                ax[row, col].set_xticks(x_ticks)
            if y_ticks is not None:
                ax[row, col].set_yticks(y_ticks)


        legend_elements = [
            Line2D([0], [0], color=x_equals_y_line_colour, linestyle='-', label='x=y'),
            Line2D([0], [0], color=regression_line_colour, linestyle='-', label='Line of\nBest Fit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=it_dot_colour, markersize=10, label='Working it\n preds'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=it0_dot_colour, markersize=10, label='it0 preds') if underlay_it0 else None,
        ]

        legend_elements = [elem for elem in legend_elements if elem is not None]

        plt.tight_layout(rect=[0.08, 0.08, 0.92, 0.92])#,pad=0.1)
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.95, 0.5), ncol=1, fontsize=legend_fontsize)

        fig.text(0.5, 0.04, self.docking_column, ha='center', fontsize=title_fontsize)
        fig.text(0.04, 0.5, f"Predicted {self.docking_column}", va='center', rotation='vertical', fontsize=title_fontsize)


        if save_plot:
            path = working_dir / f"{sel_method}_{plot_filename}"
            self.logger.info(f"Saving plot to: {path}")
            plt.savefig(path, dpi=600, bbox_inches='tight')

        plt.show()

        return

    def _pairwiseSimilarity(self, fngpts_x: list, fngpts_y: list):
        """
        Description
        -----------
        Function to calculate the Tanimoto Similarity matrix between two lists of SMILES strings

        Parameters
        ----------
        fngpts_x (list)     List of molecular fingerprints
        fngpts_y (list)     List of molecular fingerprints

        Returns
        -------
        Similarity matrix for fingerprints x and y
        """

        n_fngpts_x = len(fngpts_x)
        n_fngpts_y = len(fngpts_y)

        similarities = np.zeros((n_fngpts_x, n_fngpts_y))

        for i, fp_x in enumerate(fngpts_x):
            for j, fp_y in enumerate(fngpts_y):
                similarities[i, j] = FingerprintSimilarity(fp_x, fp_y)

        return similarities

    def tanimotoHeatMap(
        self,
        smiles_a: list,
        smiles_b: list,
        molids_a: list,
        molids_b: list,
        save_plots: bool = False,
        plot_fname: str="tanimoto_heatmap"
    ):
        """
        Description
        -----------
        Function which takes 2 lists of smiles as inputs and plots the Tanimoto Similarities.
        This analyses both within and across the two lists giving a comprehensive look into the 
        structural similarities

        Parameters
        ----------
        smiles_a (list)         list of SMILES strings
        smiles_b (list)         list of SMILES strings
        molids_a (list)         list of molecule IDs for labelling axes
        molids_b (list)         list of molecule IDs for labelling axes
        save_plots (bool)       flag to save the Tanimoto Similarity heat plots

        Returns
        -------
        Figure containing 3 heat plots:
            1- Tanimoto Similarity between SMILES in smiles_a
            2- Tanimoto Similarity between SMILES in smiles_b
            3- Tanimoto Similarity between SMILES in smiles_a and smiles_b
        """

        self.logger.info("Generating tanimoto heat map...")

        mols_a = [Chem.MolFromSmiles(smi) for smi in smiles_a]
        mols_b = [Chem.MolFromSmiles(smi) for smi in smiles_b]

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        fngpts_a = [rdkit_gen.GetFingerprint(mol) for mol in mols_a]

        fngpts_b = [rdkit_gen.GetFingerprint(mol) for mol in mols_b]

        sim_a = self._pairwiseSimilarity(fngpts_x=fngpts_a, fngpts_y=fngpts_a)

        sim_b = self._pairwiseSimilarity(fngpts_x=fngpts_b, fngpts_y=fngpts_b)

        sim_ab = self._pairwiseSimilarity(fngpts_x=fngpts_a, fngpts_y=fngpts_b)

        def heatmap(sim, x_labels, y_labels, ax):
            plot = sns.heatmap(
                sim,
                annot=True,
                annot_kws={"fontsize": 10},
                cmap="crest",
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                cbar=False,
            )

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        heatmap(sim=sim_a, x_labels=molids_a, y_labels=molids_a, ax=axes[0])
        axes[0].set_title("Heatmap Smiles A")

        heatmap(sim=sim_b, x_labels=molids_b, y_labels=molids_b, ax=axes[1])
        axes[1].set_title("Heatmap Smiles B")

        heatmap(sim=sim_ab, x_labels=molids_a, y_labels=molids_b, ax=axes[2])
        axes[2].set_title("Heatmap Smiles A vs Smiles B")

        cbar = fig.colorbar(
            axes[0].collections[0],
            ax=axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        cbar.set_label("Tanimoto Similarity")

        if save_plots:
            plt.savefig(self.plot_dir / f"{plot_fname}.png", dpi=(600))

        plt.show()

    def avgTanimotoAcrossIterations(
        self,
        experiments: list,
        smiles_df: Path = PROJ_DIR / "datasets" / "PyMolGen" / "desc"/ "rdkit" / "full_data" / "PMG_rdkit_*.csv",
        prefix: str = "PMG-",
        results_dir: Path = RESULTS_DIR,
        save_plot: bool = True,
        filename: str = "Avg_Tanimoto_Plot",
        batch_size: int=50
    ):
        """
        Dictionary
        ----------
        Function to calculate the average pairwise Tanimoto Similarity of the added training molecules
        for each experiment provided and plot them.

        Parameters
        ----------
        experiments (list)          List of experiment names (name of directories results are in)
        smiles_df (str)             Generic pathway to the .csv file containing all of the SMILES
                                    data (uses glob, e.g., /path/to/file/smiles_df_* )
        results_dir (str)           Pathway to results directory where the experiment directories are held
        save_plot (bool)            Flag to save generated plots
        save_path (str)             Pathway to directory you want to save the plots in
        filename (str)              Name of the file to save plots as
        """
        self.logger.info("Calculating averave tanimoto similarity across iterations...")

        plt.figure(figsize=(10, 6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        tan_sim_dict = {}

        for i, exp in tqdm(
            enumerate(experiments), desc="Processing Experiments", unit="exp"
        ):
            experiment_dir = results_dir + exp
            n_iters = countIterations(experiment_dir)

            all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])

            with open(Path(exp) / "run_params.json", "r") as f:
                param_dict = json.load(f)
            step = int(param_dict["batch_size"]) 

            avg_tanimoto_sim_ls = []
            iter_ls = []
            n_mols_chosen = []

            for iter in range(0, n_iters + 1):
                temp_df = pd.DataFrame()
                start_iter = iter
                end_iter = iter + 1

                molids = getMolsBetweenIters(
                    experiment_dir=experiment_dir,
                    start_iter=start_iter,
                    end_iter=end_iter,
                )
                temp_df["ID"] = molids

                smiles = molidList2Smiles(
                    molids=molids, prefix=prefix, data_fpath=smiles_df
                )
                temp_df["SMILES"] = smiles

                mols = [Chem.MolFromSmiles(smi) for smi in smiles]
                temp_df["Mols"] = mols

                added_fngpts = [rdkit_gen.GetFingerprint(mol) for mol in mols]
                temp_df["Fingerprints"] = added_fngpts

                all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
                iter_ls.append(end_iter)
                n_mols_chosen.append(end_iter * step)

                sim = self._pairwiseSimilarity(
                    fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
                )

                avg_sim = round(np.mean(sim), 4)
                avg_tanimoto_sim_ls.append(avg_sim)
                tan_sim_dict[exp] = avg_tanimoto_sim_ls

            sns.lineplot(
                x=n_mols_chosen, y=avg_tanimoto_sim_ls, label=exp, color=colours[i]
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average Tanimoro Similarity")
        plt.ylim(0, 1)
        plt.title("Average Tanimoto Similarity of Chosen Mols")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.plot_dir / f"{filename}.png", dpi=600)

        plt.show()

        return tan_sim_dict

    def topPredictionAnalysis(
        self,
        experiments: list,
        preds_fname: str = "all_preds*",
        results_dir: Path = RESULTS_DIR,
        preds_column: str = "pred_Affinity(kcal/mol)",
        n_mols: int = 1000,
        save_plot: bool = True,
        filename: str = "Avg_Top_Preds_Plot",
        sort_by_descending: bool = True,
    ):

        ascending = False if sort_by_descending else True

        for exp in tqdm(experiments, desc="Processing Experiments", unit="exp"):
            avg_top_preds = []
            n_mols_chosen = []
            n_iters = countIterations(results_dir / str(exp))

            with open(Path(exp) / "run_params.json", "r") as f:
                param_dict = json.load(f)
            step = int(param_dict["batch_size"])

            linestyle = self.linestyles["_50_"] if "_50_" in exp else self.linestyles["_10_"]
            method = next((m for m in self.method_colour_map.keys() if exp.endswith(m)), None)
            colour = self.method_colour_map.get(method, "black")

            chosen_mols = pd.read_csv(
                results_dir / exp / "chosen_mol.csv", index_col="ID"
            ).index

            for iter in range(0, n_iters + 1):
                working_dir = results_dir / exp / f"it{iter}"
                self.logger.debug(f"Working directory: {working_dir}")
                preds_files = glob(str(working_dir / preds_fname))
                top_preds_ls = []

                for file in preds_files:
                    preds = pd.read_csv(file)
                    top_preds = getTop(preds, n_mols, preds_column, ascending)
                    top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                    top_preds_ls.extend(top_preds[preds_column].tolist())

                top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
                abs_top_preds = getTop(top_preds_df, n_mols, preds_column, ascending)

                avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
                n_mols_chosen.append(iter * step)

            sns.lineplot(
                x=n_mols_chosen,
                y=avg_top_preds,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )

        plt.xlabel("Number of Molecules")
        plt.ylabel(f"Average {preds_column}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.plot_dir / f"{filename}.png", dpi=600, bbox_inches="tight")

        plt.show()

    def _calcAvgTanimotoExp(self, exp, results_dir, smiles_df, prefix):
        experiment_dir = results_dir / exp
        n_iters = countIterations(experiment_dir)

        with open(Path(experiment_dir) / "run_params.json", "r") as f:
            param_dict = json.load(f)
        step = int(param_dict["batch_size"]) 

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])
        n_mols_chosen = []
        avg_tanimoto_sim_ls = []

        for iter in range(0, n_iters + 1):
            temp_df = pd.DataFrame()
            start_iter = iter
            end_iter = iter + 1

            molids = getMolsBetweenIters(
                experiment_dir=experiment_dir, start_iter=start_iter, end_iter=end_iter
            )

            temp_df["ID"] = molids

            temp_df["SMILES"] = molidList2Smiles(
                molids=molids, prefix=prefix, data_fpath=smiles_df
            )
            temp_df["Mols"] = [Chem.MolFromSmiles(smi) for smi in temp_df["SMILES"]]

            temp_df["Fingerprints"] = [
                rdkit_gen.GetFingerprint(mol) for mol in temp_df["Mols"]
            ]

            all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
            n_mols_chosen.append(end_iter * step)

            sim = self._pairwiseSimilarity(
                fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
            )

            avg_sim = round(np.mean(sim), 4)
            avg_tanimoto_sim_ls.append(avg_sim)

        return exp, avg_tanimoto_sim_ls, n_mols_chosen, step

    def _processTopPredsExp(
        self,
        exp: str,
        preds_fname: str,
        preds_column: str,
        n_mols: int,
        ascending: bool,
    ):

        avg_top_preds = []
        n_mols_chosen = []
        experiment_dir = Path(self.results_dir) / exp
        n_iters = countIterations(self.results_dir + exp)

        with open(Path(experiment_dir) / "run_params.json", "r") as f:
            param_dict = json.load(f)
        step = int(param_dict["batch_size"]) 

        linestyle = self.linestyles["_50_"] if "_50_" in exp else self.linestyles["_10_"]
        method = next((m for m in self.method_colour_map.keys() if exp.endswith(m)), None)
        colour = self.method_colour_map.get(method, "black")
        chosen_mols = pd.read_csv(
            self.results_dir / exp / "chosen_mol.csv", index_col="ID"
        ).index

        for iter in range(0, n_iters + 1):
            working_dir = self.results_dir / exp / f"it{iter}"
            preds_files = glob(str(working_dir / preds_fname))
            top_preds_ls = []

            for file in preds_files:
                preds = pd.read_csv(file)
                top_preds = getTop(preds, n_mols, preds_column, ascending)
                top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                top_preds_ls.extend(top_preds[preds_column].tolist())

            top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
            abs_top_preds = getTop(top_preds_df, n_mols, preds_column, ascending)
            avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
            n_mols_chosen.append(iter * step)

        return exp, n_mols_chosen, avg_top_preds, colour, linestyle

    def plotMPOPotencyCorrelation(self,
                                     full_data_fpath: str,
                                     preds_df_fpath: str,
                                     save_plot: bool=False,
                                     save_fname: str='mpo_aff_correlation',
                                     dpi: int=600):
        
        """
        Description
        -----------
        Function to plot the correlation between MPO and Docking Scores

        Parameters
        ----------
        full_data_fpath (str)       Path to the full data which includes PFI and oe_logp
        preds_df_fpath (str)        Path to the Docking Scores or Predictions
        save_plot (bool)            Flag to save plot
        save_fname (str)            Name to save plot under
        save_fpath (str)            Path to save plot to
        dpi (int)                   Image quality

        Returns
        -------
        None
        
        """
        
        desc_df = pd.read_csv(
            full_data_fpath, index_col="ID", usecols=['ID', 'PFI', 'oe_logp']
        )

        preds_df = pd.read_csv(
            preds_df_fpath, index_col='ID'
        )

        mpo_df = pd.DataFrame()
        mpo_df.index = preds_df.index
        mpo_df['MPO'] = [
            -score * (1 / (1 + math.exp(PFI - 8)))
            for score, PFI in zip(preds_df[self.docking_column], desc_df["PFI"])
        ]
        mpo_df[self.docking_column] = preds_df[self.docking_column]

        sns.scatterplot(x= 'MPO', y= self.docking_column, data=mpo_df)
        plt.title(f"MPO & {self.docking_column} Correlation")
        plt.xlabel('MPO')
        plt.ylabel(self.docking_column)

        # Add line of best fit using regplot
        sns.regplot(
            data=mpo_df,
            x="MPO",
            y=self.docking_column,
            scatter=False,
            line_kws={"linestyle": "-", "color": 'gold'},
        )

        slope, intercept = np.polyfit(mpo_df['MPO'], mpo_df[self.docking_column], 1)

        plt.text(
            x=mpo_df['MPO'].max() * 0.75,  # Position on x-axis
            y=mpo_df[self.docking_column].max(),  # Position on y-axis
            s=f"$y = {slope:.2f}x + {intercept:.2f}$",
            fontsize=12,
            color='black',
        )

        r2 = r2_score(mpo_df['MPO'], mpo_df[self.docking_column])
        r_pearson, p_pearson = pearsonr(mpo_df['MPO'], mpo_df[self.docking_column])

        plt.text(
            x=mpo_df['MPO'].max() * 0.75,
            y=mpo_df[self.docking_column].max() * 1.05,
            s = f"$pearson r = {r_pearson: .2f}$",
            fontsize=12,
            color='black',
        )

        if save_plot:
            plt.savefig(self.plot_dir / f"{save_fname}.png", dpi=dpi)

        plt.show()
        
        return
    
    def drawChosenMols(self,
                         experiment: str,
                         iter_ls: list,
                         save_img: bool=False,
                         img_fname: str=None,
                         full_data_fpath: str=None):
        
        """
        Description
        -----------
        Function to take chosen mols at specified iterations, draw their structure 
        and save as a .png file.

        Parameters
        ----------
        experiment (str)        name of experiment e.g., 202410002_10_mp
        iter_ls (list)          List of iterations from which to obtain the chosen molecules
                                from
        save_img (bool)         Flag to save image of drawn molecules
        img_fpath (str)         Path to save image to
        img_fname (str)         Name to save image under

        Returns
        -------
        Image of drawn chosen molecules

        """
        
        chosen_mol_df = pd.read_csv(self.results_dir / experiment / 'chosen_mol.csv', index_col='ID')
        trimmed_df = chosen_mol_df[chosen_mol_df['Iteration'].isin(iter_ls)]

        trimmed_molids = trimmed_df.index
        iteration_ls = trimmed_df['Iteration'].tolist()
        iteration_legend = [f"Iter {n}" for n in iteration_ls]

        smiles_ls = molidList2Smiles(trimmed_molids, 'PMG-', full_data_fpath)
        mol_ls = [Chem.MolFromSmiles(smiles) for smiles in smiles_ls]

        img = Chem.Draw.MolsToGridImage(mols=mol_ls, 
                                        molsPerRow=5, 
                                        subImgSize=(200,200), 
                                        legends=iteration_legend,
                                        )          
        
        if save_img:
            try:
                img_pil = PILImage.open(BytesIO(img.data))
            except AttributeError as e:
                img_pil = img

            img_pil.save(self.plot_dir / f"{img_fname}.png")

        return img
    
    def _processIteration(self, args):
        experiment, pred_desc, n_components, indi_plot_suffix, remove_outliers, iter, full_fpath, chembl_fpath= args
        self.logger.debug(f"Processing Iteration {iter}")
        training_data = self.results_dir / experiment / f"it{iter}"/ "training_data" / "training_targets.csv.gz"
        molid_ls = pd.read_csv(training_data, index_col='ID').index
        desc_df = getDesc4Molid(molid_ls, pmg_fpath=full_fpath, chembl_fpath=chembl_fpath, logger=self.logger)

        self.logger.debug(f"Full descriptor data frame:\n{pred_desc}")
        self.logger.debug(f"Training descriptor data frame:\n{desc_df}")


        pca_plot = self.plotPCA(
                    train = desc_df,
                    prediction=pred_desc,
                    source_ls=[
                        'Train',
                        'PyMolGen'
                    ],
                    n_components=n_components,
                    save_plot=True,
                    plot_area=True,
                    plot_scatter=True,
                    plot_fname=f'PCA_{experiment}_iter_{iter}_{indi_plot_suffix}',
                    plot_title=f"{experiment} Iteration {iter}",
                    remove_outliers=remove_outliers
                    )
        plt_img = fig2Img(pca_plot)

        return (iter, plt_img)
            
    def plotPCAAcrossIters(self,
                           experiment: str,
                           iter_ls: list,
                           prediction_descs: str,
                           chembl_descs:str,
                           n_components: int=2,
                           n_rows: int=2,
                           save_plot: bool=False,
                           indi_plot_suffix: str=None,
                           plot_fname: str="PCA_Across_Iters",
                           plot_in_one_fig: bool=False,
                           remove_outliers:bool=True,
                           use_multiprocessing: bool=False,
                              ):
        
        """
        Description
        -----------
        Function to plot a grid image of PCA plots across iterations, showing the development
        of the overlap between training data and full data

        Parameters
        ----------
        experiment (str)        Name of the experiment e.g., 20241002_10_mp
        iter_ls (list)          List of iterations to plot
        n_components (int)      Number of principal components to plot
        n_rows (int)            Number of rows and columns to make a NxN grid of PCA plots
        save_plot (bool)        Flag to save plot
        save_fpath (str)        Path to save plot to
        plot_fname (str)        Name to save plot as

        
        Returns
        -------
        None
        """

        process_args = [(
            experiment, 
            prediction_descs, 
            n_components,
            indi_plot_suffix,
            remove_outliers,
            it,
            prediction_descs,
            chembl_descs,
        ) for it in iter_ls]

        if use_multiprocessing:
            with Pool() as pool:
                results = pool.map(self._processIteration, process_args)
        
        else:
            results = []
            for args in process_args:
                res = self._processIteration(args)
                results.append(res)

        if plot_in_one_fig:
            fig, outer_ax = plt.subplots(n_rows, n_rows, figsize=(20, 25), gridspec_kw={'wspace':0, 'hspace':0.2})
            results_dict = dict(results)

            for idx, it in enumerate(iter_ls):
                row = idx // n_rows
                col = idx % n_rows
                outer_ax[row, col].imshow(results_dict[it])
                outer_ax[row, col].axis('off')
                outer_ax[row, col].set_title(f'Iteration {it}')
            
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout(pad=0)

            if save_plot:
                plt.savefig(self.plot_dir / f'/{plot_fname}.png', dpi=300, bbox_inches='tight')

        return

    def _submitDockingWrapper(self, args):
                        
        batch_no, idxs_in_batch, docking_score_files, docking_column, docking_dir, receptor_path, max_confs = args

        if "*" in str(docking_score_files) or "?" in str(docking_score_files):
            docking_score_batch_file = str(docking_score_files).replace("*", str(batch_no))
        
        else:
            docking_score_batch_file = docking_score_files

        da = DatasetAccessor(
            original_path=docking_score_batch_file,
            wait_time=30,
            logger=self.logger
        )

        # Obtain exclusive access to the docking file
        docking_file = da.getExclusiveAccess()
        if docking_file is None:
            self.logger.debug(f"Failed to access file:\n{docking_score_batch_file}")
            self.logger.debug(f"Redocking of IDs:\n{idxs_in_batch} required")

        dock_df = pd.read_csv(docking_file, index_col=0)
        self.logger.debug(f"Printing dock_df:\n{dock_df}")

        # Isolating the molecule ids which have not already been docked or in the process of being docked
        for_docking = getUndocked(
            dock_df=dock_df,
            idxs_in_batch=idxs_in_batch,
            scores_col=docking_column,
            logger=self.logger
        )

        if for_docking.empty:
            self.logger.debug(f"No molecules to dock in batch {batch_no}...")
            da.releaseFile()
            return None, None, docking_score_batch_file, [], idxs_in_batch

        # Change docking value for each molecule being docked as 'PD' (pending)
        da.editDF(
            column_to_edit=docking_column,
            idxs_to_edit=for_docking.index,
            vals_to_enter=["PD" for idx in for_docking.index],
        )

        # Releases exclusive access on file so parallel runs can access it
        da.releaseFile()

        self.logger.info(f"** Docking compounds: {for_docking.index.tolist()}")

        molid_ls = []
        smi_ls = []

        for molid, smi in for_docking["SMILES"].items():
            molid_ls.append(molid)
            smi_ls.append(smi)

        # Initialising the docker
        docker = RunGNINA(
            docking_dir=docking_dir,
            molid_ls=molid_ls,
            smi_ls=smi_ls,
            receptor_path=receptor_path,
            max_confs=max_confs,
        )

        # Creating sdfs with numerous conformers and adjusting for pH 7.4
        docker.processMols(use_multiprocessing=True)

        # Docking the molecules and saving scores in for_docking
        job_ids = docker.submitMultipleJobs(run_hrs=0, run_mins=20, use_multiprocessing=True)

        return docker, job_ids, docking_score_batch_file, molid_ls, idxs_in_batch

    def _dockingScoreRetrieval(
        self,
        dock_scores_ls: list,
        docking_batch_file: list,
        mols_to_edit_ls: list,
        idxs_in_batch: list,
        docking_column:str='Affinity(kcal/mol)'
    ):

        da = DatasetAccessor(
            original_path=docking_batch_file,
            wait_time=30,
        )

        if mols_to_edit_ls:
            da.getExclusiveAccess()

            da.editDF(
                column_to_edit=docking_column,
                idxs_to_edit=mols_to_edit_ls,
                vals_to_enter=dock_scores_ls,
            )

            da.releaseFile()

            wait4Docking(
                docking_batch_file,
                idxs_in_batch=idxs_in_batch,
                scores_col=docking_column,
                check_interval=60,
            )

        file_accessed = False
        while not file_accessed:
            try:
                batch_dock_df = pd.read_csv(docking_batch_file, index_col=0)
                file_accessed = True
            except FileNotFoundError as e:
                self.logger.info("Waiting for file to be accessable again...")
                time.sleep(30)

        batch_dock_df = batch_dock_df.loc[idxs_in_batch]

        return batch_dock_df
     
    def dockTopPred(self,
                    experiment:str,
                    iter:int,                
                    data_fpath: str,
                    docking_dir: str,
                    receptor_path: str,
                    max_confs: int,
                    preds_fname: str='all_preds_*',
                    ascending:bool=True,
                    search_in_top:int=20,
                    prefix: str='PMG-',
   
                    chunksize:int=100000,
                    docking_column:str='Affinity(kcal/mol)',
                    preds_column:str='pred_Affinity(kcal/mol)',


    ):
        
        self.logger.info(f"Docking top predicted molecules for: {experiment}")
        path_to_preds = self.results_dir / experiment / f"it{iter}"

        if '*' in str(preds_fname) or '?' in str(preds_fname):
            self.logger.debug(f"Printing file path:\n{path_to_preds / preds_fname}")
            flist = glob(str(path_to_preds / preds_fname))
            self.logger.debug(f"Printing file list:\n{flist}")
            df_list = [pd.read_csv(file) for file in flist]
            preds_df = getTop(
                df=df_list, 
                n=search_in_top, 
                column=f'{preds_column}', 
                ascending=ascending
                )
            
        else: 
            preds_df = getTop(
                df=pd.read_csv(path_to_preds / preds_fname),
                n=search_in_top, 
                column=f'{preds_column}', 
                ascending=ascending
                )

        self.logger.info(f"Obtained top {search_in_top} predicted molecules")
        self.logger.debug(f"Printing Prediction data frame:\n{preds_df}")

        molid_ls = preds_df['ID'].tolist()

        self.logger.debug(f"Printing Molecule ID list:\n{molid_ls}")

        smi_ls = molidList2Smiles(
                molids = molid_ls,
                prefix = prefix,
                data_fpath = data_fpath
            )

        df_select = pd.DataFrame()
        df_select['ID'] = molid_ls
        df_select["batch_no"] = [
            molid2BatchNo(
                molid=molid, prefix=prefix, dataset_file=data_fpath, chunksize=chunksize
            )
            for molid in molid_ls
        ]
        df_select['SMILES'] = smi_ls

        sdw_args = [
            (
                batch_no, 
                idxs_in_batch, 
                data_fpath,
                self.docking_column,
                docking_dir,
                receptor_path,
                max_confs,
            )
            for batch_no, idxs_in_batch in (
                df_select.reset_index()
                .groupby("batch_no")['ID']
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

        for args in sdw_args:
            docker, job_ids, ds_batch_file, mols_to_edit_ls, idx_ls = (
                self._submitDockingWrapper(args)
            )
        
            if docker is not None:
                initialised_dockers.append(docker)
                all_job_id_ls.extend(job_ids)
                all_docking_score_batch_files.append(ds_batch_file)
                all_molid_ls.append(mols_to_edit_ls)
                all_idxs_in_batch.append(idx_ls)
        
            else:
                docked_df = pd.read_csv(ds_batch_file, index_col="ID")
                all_dock_scores_ls.append(docked_df[docking_column].loc[idx_ls])
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

        df = pd.DataFrame()
        df['ID'] = molid_ls
        df['SMILES'] = smi_ls
        df['Batch_No'] = df_select['batch_no']
        df[preds_column] = df.merge(preds_df[['ID', preds_column]], on='ID', how='left')[preds_column]
        df[docking_column] = np.nan


        with Pool() as pool:
            results = pool.starmap(self._dockingScoreRetrieval, dsr_args)

        fin_dock_df = pd.concat(results, axis=0)
        # Reset index and keep the old index as a column
        fin_dock_df.reset_index(drop=False, inplace=True)   

        # Make sure 'ID' exists in fin_dock_df after resetting index
        if 'ID' not in fin_dock_df.columns:
            fin_dock_df['ID'] = fin_dock_df['index']  # If the 'ID' column was not found, use the 'index' column

        # Now iterate over the index of fin_dock_df to update docking scores in df
        for id in fin_dock_df['ID']:  # Use the 'ID' column from fin_dock_df to access values
            df.loc[df['ID'] == id, docking_column] = fin_dock_df.loc[fin_dock_df['ID'] == id, docking_column].values[0]

        # Set the 'ID' column as the index in df
        df.set_index('ID', inplace=True)
        df[docking_column] = pd.to_numeric(df[docking_column], errors='coerce')
        df[preds_column] = pd.to_numeric(df[preds_column], errors='coerce')

        return df
    
    def plotTopPredDocked(self,
                            experiment_ls:list,
                            iter:int,                            
                            data_fpath: str,
                            docking_dir: str,
                            receptor_path: str,
                            max_confs: int,                             
                            preds_fname: str='all_preds_*',
                            ascending:bool=True,
                            search_in_top:int=20,
                            prefix: str='PMG-',
                            chunksize:int=100000,
                            docking_column:str='Affinity(kcal/mol)',
                            preds_column:str='pred_Affinity(kcal/mol)',
                            save_plot:bool=False,
                            plot_name:str='Top_pred_docked_boxplot',
                            save_structures: bool=True,
                            true_yticks: list=[-12, -10.5, -9, -7.5],
                            pred_yticks: list=[-12, -10.5, -9, -7.5],
                            tick_fontsize: int=18,
                            label_fontsize:int=18,
                            legend_fontsize: int=16,
                            method_legend_map: dict=None
                    ):
                
        """
        Description
        -----------
        Function to plot the docking scores of the top predicted molecules for specified experiments
        
        Parameters
        ----------
        experiment_ls
        iter
        preds_fname
        ascending
        search_in_top
        prefix
        data_fpath
        chunksize
        docking_column
        preds_column

        Returns
        -------
        
        """


        df_list = []
        stats_df_list = []

        for exp in experiment_ls:
            self.logger.info(f"Processing {exp}")

            self.logger.debug(
                "Arguments into dockTopPred:\n"
                f"  Experiment: {exp}\n"
                f"  Iteration: {iter}\n"
                f"  Docking Dir: {docking_dir}\n"
                f"  Receptor Path: {receptor_path}\n"
                f"  Max Conformers: {max_confs}\n"
                f"  Predictions Filename: {preds_fname}\n"
                f"  Ascending: {ascending}\n"
                f"  Search in Top: {search_in_top}\n"
                f"  Prefix: {prefix}\n"
                f"  Data File Path: {data_fpath}\n"
                f"  Chunk Size: {chunksize}\n"
                f"  Docking Column: {docking_column}\n"
                f"  Predictions Column: {preds_column}"
            )

            exp_docking_df = self.dockTopPred(
                            experiment=exp,
                            iter=iter,
                            docking_dir=docking_dir,
                            receptor_path=receptor_path,
                            max_confs=max_confs,   
                            preds_fname=preds_fname,
                            ascending=ascending,
                            search_in_top=search_in_top,
                            prefix=prefix,
                            data_fpath=data_fpath,
                            chunksize=chunksize,
                            docking_column=docking_column,
                            preds_column=preds_column
                            )
        
            exp_docking_df['Experiment'] = exp
            df_list.append(exp_docking_df)
            stats_data = {
            "Experiment" : exp,
            'Mean Docking Score': exp_docking_df[docking_column].mean(),
            'Min Docking Score': exp_docking_df[docking_column].min(),
            'Max Docking Score': exp_docking_df[docking_column].max(),            
            'Std Dev Docking Score': exp_docking_df[docking_column].std(),
            'Mean Predicted Score': exp_docking_df[preds_column].mean(),
            'Min Pred_Docking Score': exp_docking_df[preds_column].min(),
            'Max Pred_Docking Score': exp_docking_df[preds_column].max(),            
            'Std Dev Predicted Score': exp_docking_df[preds_column].std()
        }
            self.logger.debug(f"Printing stats_data:\n{stats_data}")
            stats_df = pd.DataFrame([stats_data])
            stats_df_list.append(stats_df)


        
        full_df = pd.concat(df_list, axis=0)
        full_stats_df = pd.concat(stats_df_list, axis=0)
        full_df = full_df.reset_index()

        experiment_colors = [
            next((color for key, color in self.method_colour_map.items() if exp.endswith(key)), "gray")
            for exp in experiment_ls
        ]      

        fig = plt.figure(figsize=(12,8))
        gs = gridspec.GridSpec(2,2, width_ratios=[1,2], height_ratios=[1,1])
        
        ax2 = fig.add_subplot(gs[1,0])
        ax1 = fig.add_subplot(gs[0,0], sharex=ax2)
        ax3 = fig.add_subplot(gs[:,1])

        
        # Convert docking_column to numeric
        full_df[docking_column] = pd.to_numeric(full_df[docking_column], errors='coerce')

        # Convert preds_column to numeric
        full_df[preds_column] = pd.to_numeric(full_df[preds_column], errors='coerce')

        sns.boxplot(
            y=full_df[docking_column], x=full_df['Experiment'], ax=ax1, 
            palette=experiment_colors, legend=False, hue=None
            )
        sns.boxplot(
            y=full_df[preds_column], x=full_df['Experiment'], ax=ax2, 
            palette=experiment_colors, legend=False, hue=None
            )
        
        # Set y-ticks for both boxplot axes
        min_value = min(full_df[docking_column].min(), full_df[preds_column].min())
        max_value = max(full_df[docking_column].max(), full_df[preds_column].max())

        min_value = np.floor(min_value)
        max_value = np.ceil(max_value)

        ax1.set_yticks(true_yticks)
        ax2.set_yticks(pred_yticks)
    

        ax1.set_ylim([min_value, max_value])
        ax2.set_ylim([min_value, max_value])

        exp_names = [e.split("_")[-1] for e in experiment_ls]
        tick_positions = range(len(exp_names))
        ax2.set_xticks(tick_positions)  # Set tick positions
        
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', labelbottom=False)  # Keep ticks, just hide the text
        ax1.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.set_xticklabels(exp_names, rotation=45, fontsize=tick_fontsize)

        ax1.set_ylabel(docking_column, fontsize=label_fontsize)
        ax2.set_ylabel(f"Predicted {docking_column}", fontsize=label_fontsize)

        ax1.set_xlabel("")
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=label_fontsize)

        # Formatting Scatterplot
        sns.scatterplot(y=full_df[preds_column], x=full_df[docking_column], ax=ax3, 
                     palette=experiment_colors, hue=full_df['Experiment']
                     )
        ax3.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--', label='x=y')
        ax3.set_xticks(true_yticks)
        ax3.set_yticks(pred_yticks)
        ax3.tick_params(axis='both', labelsize=label_fontsize)
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_xlabel(docking_column, fontsize=label_fontsize)
        ax3.set_ylabel(f"Predicted {docking_column}", fontsize=label_fontsize)

        # Determine which suffixes are actually used
        used_suffixes = set()
        for exp in experiment_ls:
            matched_suffix = next((s for s in self.method_colour_map if exp.endswith(s)), None)
            if matched_suffix:
                used_suffixes.add(matched_suffix)

        # Build labels and handles in user-defined order
        legend_handles = []
        legend_labels = []

        if method_legend_map:
            for suffix, label in method_legend_map.items():
                if suffix in used_suffixes:
                    color = self.method_colour_map.get(suffix)
                    if color:
                        legend_handles.append(Line2D([0], [0], color=color, lw=2))
                        legend_labels.append(label)
                    else:
                        self.logger.warning(f"Warning: No color found for suffix {suffix}")

        # For each experiment (hue), calculate and plot the line of best fit
        for experiment, label in zip(full_df['Experiment'].unique(), legend_labels):
            # Filter the data by the current experiment
            data = full_df[full_df['Experiment'] == experiment]
            
            # Perform linear regression (slope, intercept, r_value, p_value, std_err)
            slope, intercept, _, _, _ = linregress(data[docking_column], data[preds_column])
            
            # Calculate the range of the x and y values for this subset of the data
            x_min, x_max = data[docking_column].min(), data[docking_column].max()
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            
            # Plot the line of best fit for this experiment (only within the data range)
            ax3.plot([x_min, x_max], [y_min, y_max], color=self.method_colour_map[label], lw=2)

        ax3.legend(handles=legend_handles,
                labels=legend_labels,
                title="Experiment",
                loc='best',
                fontsize=legend_fontsize,
                title_fontsize=label_fontsize)

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.plot_dir / f"{plot_name}.png")
            full_stats_df.to_csv(self.plot_dir / f"{plot_name}_stats.csv", index=False)


        plt.show()

        if save_structures:
            for df, exp in zip(df_list, experiment_ls):
                mol_ls = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
                drawn_mols = Draw.MolsToGridImage(mols=mol_ls, 
                                        molsPerRow=5, 
                                        subImgSize=(200,200),
                                        legends=df.index.astype(str).tolist(),
                                        useSVG=False)
                # drawn_mols = Image.fromarray(drawn_mols)
                df.to_csv(self.plot_dir / f"{exp}_{search_in_top}_it{iter}_structs.csv", index_label='ID')
                
                # Save the image
                drawn_mols.save(self.plot_dir / f"{exp}_{search_in_top}_it{iter}_structs.png")
    
    def uncertaintyChecker(self,
                           experiment_ls:list,
                            iter_ls: list,
                            n_plots: int=16,
                            prediction_fpath: str = "held_out_preds.csv",
                            true_path: str = PROJ_DIR / "datasets" / "held_out_data" / "PMG_held_out_targ_trimmed.csv",
                            dot_size: int = 10,
                            save_plot: bool=True,
                            plot_name: str = "preds_dev_plot.png",
                            tick_fontsize: int=18,
                            label_fontsize: int=20,
                            legend_fontsize: int=20,
                            figsize:tuple=(14, 14),
                            ):
        
        n_y_plots = int(np.sqrt(n_plots))
        n_x_plots = n_y_plots
        
        # Initialising subplots
        fig, axarr = plt.subplots(nrows=n_x_plots, ncols=n_y_plots, figsize=figsize, sharex=True, sharey=True)
        axarr = axarr.flatten()
        
        true_df = pd.read_csv(true_path, index_col='ID')
        true_df = true_df[['Affinity(kcal/mol)']].astype(float)

        for i, it in enumerate(iter_ls):
            ax = axarr[i]
            ax.set_title(f"{it * 50} mols", fontsize=label_fontsize)

            for exp in experiment_ls:
                exp_name = exp.split("_")[-1]
                suffix = "_" + exp_name
                dot_color = self.method_colour_map.get(suffix, "gray")

                working_dir = f"{self.results_dir}/{exp}"
                pred_path = f"{working_dir}/it{it}/{prediction_fpath}"

                pred_df = pd.read_csv(pred_path, index_col='ID')
                self.logger.debug(f"Printing preds path and content:\n{pred_path}\n{pred_df}")

                aligned_true = true_df.loc[pred_df.index]
                true_docking = aligned_true['Affinity(kcal/mol)']
                pred_docking = pred_df['pred_Affinity(kcal/mol)']
                uncertainty = pred_df['Uncertainty']
                error = np.abs(true_docking - pred_docking)

                sorted_idx = np.argsort(uncertainty)
                uncertainty = uncertainty.iloc[sorted_idx]
                error = error.iloc[sorted_idx]

                ax.scatter(uncertainty, error, label=exp_name, alpha=0.2, s=dot_size, color=dot_color)

        for ax in axarr:
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=tick_fontsize)

        # Global axis labels
        fig.text(0.5, 0.02, "Uncertainty", ha='center', fontsize=label_fontsize)
        fig.text(0.02, 0.5, "Prediction Error", va='center', rotation='vertical', fontsize=label_fontsize)

        # Get unique experiment suffixes
        legend_elements = []
        used_labels = set()

        for exp in experiment_ls:
            exp_name = exp.split("_")[-1]
            suffix = "_" + exp_name
            color = self.method_colour_map.get(suffix, "gray")

            if exp_name not in used_labels:
                legend_elements.append(Line2D(
                    [0], [0], marker='o', color='w', markerfacecolor=color,
                    markersize=8, label=exp_name, alpha=1.0))
                used_labels.add(exp_name)

        fig.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=legend_fontsize,
            title="Experiments",
            title_fontsize=label_fontsize,
            handlelength=2,
            markerscale=2
        )

        # Adjust layout for global labels and legend
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Left, Bottom, Right, Top
        
        if save_plot:
            plt.savefig(self.plot_dir / f"{plot_name}.png", dpi=600)


    def plotFeatImportanceAndRidgelines(
        self,
        experiment: str,
        iter_ls: list,
        importance_fpath: str = "/feature_importance_df.csv",
        top_n_feats: int = 10,
        save_data: bool = True,
        filename: str = "feature_importances_and_ridgelines",
        dpi: int = 500,
        tick_fontsize: int = 20,
        label_fontsize: int = 24,
        exp_suffix=None,
        extra_sources: dict = {}
    ):
        save_path = self.plot_dir / 'feature_ridgeline_plots'
        if not save_path.exists():
            save_path.mkdir()

        external_styles = {
            "PyMolGen": {"facecolor": "#000000", "hatch": "."},
            "ChEMBL": {"facecolor": "#000000", "hatch":"-"},
            "Hits": {"facecolor": "#000000", "hatch": "+"},
        }

        all_data = []

        if not exp_suffix:
            exp_suffix = experiment[7:]

        # Step 1: Load top-N feature importances from final iteration
        final_iter = max(iter_ls)
        imp_path = f"{self.results_dir}/{experiment}/it{final_iter}{importance_fpath}"
        try:
            feat_importance_df = pd.read_csv(imp_path)
        except Exception as e:
            self.logger.error(f"Could not read importance file for iteration {final_iter}: {e}")
            return

        feat_importance_df = feat_importance_df.sort_values("Importance", ascending=False)
        top_feats = feat_importance_df.head(top_n_feats).copy()
        features = top_feats["Feature"].tolist()

        # Step 2: Create consistent color map
        palette = sns.color_palette("tab10", n_colors=top_n_feats)
        color_map = dict(zip(features, palette))

        # Step 3: Barplot of feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=top_feats,
            x="Importance",
            y="Feature",
            palette=color_map,
            dodge=False,
            hue="Feature",
            legend=False,
        )
        plt.xlabel("Importance", fontsize=label_fontsize)
        plt.ylabel("Feature", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize, rotation=45)
        plt.yticks(fontsize=tick_fontsize)

        plt.tight_layout()

        if save_data:
            plt.savefig(save_path / f"{filename}_importance{exp_suffix}_barplot.png", dpi=dpi)
            feat_importance_df.to_csv(save_path / f"feature_importance_df{exp_suffix}.csv")

        plt.show()

        for it in iter_ls:
            if it == 0:
                continue

            glob_path = self.results_dir / f"*{exp_suffix}" /f"it{it}" / "training_data" / "training_features.csv.gz"
            training_files = glob(str(glob_path))

            for f in training_files:
                try:
                    df = pd.read_csv(f)
                    df = df[df['ID'].astype(str).str.startswith('PMG-')]
                    sub_df = df[features].copy()
                    sub_df["iteration"] = it
                    sub_df["source"] = Path(f).parent.parent.name
                    all_data.append(sub_df)
                except Exception as e:
                    self.logger.error(f"Failed to read {f}: {e}")

        if not all_data:
            self.logger.error("No valid training data found.")
            return

        combined_df = pd.concat(all_data)

        # Step 4b: Preload all PyMolGen data for use in each feature loop
        pmg_files = glob(str(PROJ_DIR / "datasets" / "PyMolGen" / "desc" / "rdkit" /"PMG_rdkit_desc_*.csv"))
        pmg_raw_data = {}
        for file in pmg_files:
            try:
                df = pd.read_csv(file)
                for col in df.columns:
                    if col not in pmg_raw_data:
                        pmg_raw_data[col] = []
                    pmg_raw_data[col].append(df[col].dropna())
            except Exception as e:
                self.logger.error(f"Failed to read {file}: {e}")

        extra_raw_data = {
            "ChEMBL": {},
            "Hits": {}
        }

        for label, path in extra_sources.items():
            try:
                df = pd.read_csv(path)
                for col in features:
                    if col in df.columns:
                        extra_raw_data[label].setdefault(col, []).append(df[col].dropna())
            except Exception as e:
                self.logger.error(f"Failed to read {label} data from {path}: {e}")

        # Step 5: Ridgeline plots for each feature
        for feat in features:
            df_plot = combined_df[[feat, "iteration"]].copy()
            df_plot = df_plot.rename(columns={feat: "value"})
            df_plot = df_plot.dropna(subset=["value"])

            if feat in pmg_raw_data:
                all_values = pd.concat(pmg_raw_data[feat]).dropna()
                all_values = pd.to_numeric(all_values, errors='coerce').dropna()
                if not all_values.empty:
                    pmg_df = pd.DataFrame({"value": all_values, "iteration": "PyMolGen"})
                    df_plot = pd.concat([df_plot, pmg_df], ignore_index=True)
                else:
                    self.logger.error(f"No PyMolGen data for feature: {feat}")
            else:
                self.logger.error(f"{feat} not found in PyMolGen descriptors")

            # Add ChEMBL and Hits if available
            for label, feat_dict in extra_raw_data.items():
                if feat in feat_dict:
                    all_values = pd.concat(feat_dict[feat]).dropna()
                    all_values = pd.to_numeric(all_values, errors="coerce").dropna()
                    if not all_values.empty:
                        label_df = pd.DataFrame({"value": all_values, "iteration": label})
                        df_plot = pd.concat([df_plot, label_df], ignore_index=True)
                    else:
                        self.logger.error(f"No valid {label} data for feature: {feat}")

            if df_plot.empty or df_plot["value"].nunique() <= 1:
                self.logger.warning(f"Skipping {feat}: no variation or all NaNs.")
                continue

            ordered_iters = ["PyMolGen", "ChEMBL", "Hits"] + [str(it) for it in sorted(iter_ls) if it != 0]

            df_plot["iteration"] = pd.Categorical(
                df_plot["iteration"].astype(str),
                categories=ordered_iters,
                ordered=True
            )

            df_plot = df_plot.sort_values("iteration")

            for group in df_plot["iteration"].unique():
                group_mask = df_plot["iteration"] == group
                values = df_plot.loc[group_mask, "value"]
                if values.nunique() == 1:
                    scale = values.abs() * 0.00001
                    jitter_std = np.maximum(scale, 0.001)
                    df_plot.loc[group_mask, "value"] += np.random.normal(0, jitter_std, size=len(values))

            fig, ax = plt.subplots(figsize=(10, 6))

            y_ticks = []
            y_labels = []
            x_vals = np.linspace(df_plot["value"].min(), df_plot["value"].max(), 500)

            for i, it in enumerate(ordered_iters):
                subset = df_plot[df_plot["iteration"] == it]["value"]
                if subset.empty:
                    continue

                values = subset.values

                if np.unique(values).size == 1:
                    val = values[0]
                    bump_y = np.exp(-0.5 * ((x_vals - val) / 0.2) ** 2)
                    bump_y = bump_y / bump_y.max() * 0.9

                    style = external_styles.get(it, {})
                    facecolor = style.get("facecolor", color_map[feat])
                    hatch = style.get("hatch", None)

                    ax.fill_between(
                        x_vals, i, i + bump_y,
                        facecolor=facecolor,
                        edgecolor="black",
                        hatch=hatch,
                        linewidth=0.5
                    )
                    ax.plot(x_vals, i + bump_y, color="black", linewidth=1)

                else:
                    kde = gaussian_kde(values)
                    y_vals = kde(x_vals)
                    y_scaled = y_vals / y_vals.max() * 0.9

                    style = external_styles.get(it, {})
                    facecolor = style.get("facecolor", color_map[feat])
                    hatch = style.get("hatch", None)

                    ax.fill_between(
                        x_vals, i, i + y_scaled,
                        facecolor=facecolor,
                        edgecolor="black",
                        hatch=hatch,
                        linewidth=0.5,
                        alpha=0.7 if it not in external_styles else 1.0
                    )
                    ax.plot(x_vals, i + y_scaled, color="black", linewidth=1)


                y_ticks.append(i + 0.5)
                y_labels.append(str(it))

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=tick_fontsize)
            ax.set_xlabel(feat, fontsize=label_fontsize)
            ax.set_ylabel("Iteration", fontsize=label_fontsize)
            ax.tick_params(axis="x", labelsize=tick_fontsize)
            plt.tight_layout()

            if save_data:
                ridgeline_path = save_path / f"{feat.replace('/', '_')}_ridgeline{exp_suffix}.png"
                plt.savefig(ridgeline_path, dpi=dpi)

            plt.show()

    def plotFeatureLoadingEvolution(self,
                                    experiment: str,
                                    iter_ls: list,
                                    importance_fpath: str = "feature_importance_df.csv",
                                    top_n_feats: int = 10,
                                    save_data: bool = False,
                                    filename: str = "feature_loading_shift_subplots",
                                    dpi: int = 500,
                                    tick_fontsize: int = 14,
                                    label_fontsize: int = 16,
                                    title_fontsize: int = 18):

        save_path = self.plot_dir / "feature_loading_evolution"
        if not save_path.exists():
            save_path.mkdir()

        exp_suffix = experiment[7:]

        # Build importance matrix
        importance_df = pd.DataFrame()
        for it in iter_ls:
            path = (RESULTS_DIR / "init_RF_model" / "it0" / "feature_importance_df.csv"
                    if it == 0 else RESULTS_DIR / experiment / f"it{it}" / importance_fpath)
            df = pd.read_csv(path, usecols=["Feature", "Importance"]).set_index("Feature")
            df.columns = [f"it{it}"]
            importance_df = importance_df.join(df, how="outer")

        importance_df = importance_df.fillna(0)

        # Top N features
        top_feats = importance_df.mean(axis=1).sort_values(ascending=False).head(top_n_feats).index.tolist()

        # Collect loadings
        feature_loadings = {feat: [] for feat in top_feats}

        for i, current_iter in enumerate(iter_ls):
            path = (RESULTS_DIR / "init_RF_model" / "it0" / "feature_importance_df.csv"
                    if current_iter == 0 else RESULTS_DIR / experiment / f"it{current_iter}" / importance_fpath)

            df = pd.read_csv(path, usecols=["Feature", "Importance"])
            df = df[df["Feature"].isin(top_feats)].set_index("Feature").reindex(top_feats).fillna(0)
            row = df.values.flatten()
            noisy_matrix = row + np.random.normal(0, 1e-6, size=(5, len(row)))

            pca = PCA(n_components=2)
            pca.fit(noisy_matrix)
            explained_var = pca.explained_variance_ratio_ * 100
            loadings = pca.components_.T

            for j, feat in enumerate(top_feats):
                feature_loadings[feat].append((loadings[j], explained_var))

        # Plot
        n_cols = 3
        n_rows = int(np.ceil(len(top_feats) / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), gridspec_kw={'wspace':0.4, 'hspace':0.4})
        axs = axs.flatten()
        colors = sns.color_palette("tab10", n_colors=len(top_feats))

        for idx, (feat, color) in enumerate(zip(top_feats, colors)):
            ax = axs[idx]
            trail = np.array([pt[0] for pt in feature_loadings[feat]])
            explained_var = feature_loadings[feat][-1][1]

            for t, (vec, _) in enumerate(feature_loadings[feat]):
                tip_x, tip_y = vec
                ax.arrow(0, 0, tip_x, tip_y,
                        head_width=0.02,
                        head_length=0.03,
                        fc=color, ec=color,
                        alpha = 0.3 + 0.7 * (t / (len(feature_loadings[feat]) - 1)),
                        linewidth=1.5,
                        length_includes_head=True)

                # Dynamically offset text away from the arrow tip (moved inside the loop)
                norm = np.linalg.norm([tip_x, tip_y])
                if norm == 0:
                    norm = 1
                unit_vec = np.array([tip_x, tip_y]) / norm
                label_pos = np.array([tip_x, tip_y]) + unit_vec * 0.08

                ax.text(label_pos[0], label_pos[1],
                        str(iter_ls[t]), fontsize=10, color='black',
                        ha='center', va='center')

            ax.axhline(0, color='gray', linestyle='--')
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_aspect('equal')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(feat, fontsize=title_fontsize)
            ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% Var)", fontsize=label_fontsize)
            ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% Var)", fontsize=label_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Remove any unused subplots
        for j in range(len(top_feats), len(axs)):
            fig.delaxes(axs[j])

        # Add legend for feature colors (placed outside on the right)
        legend_patches = [
            plt.Line2D([0], [0], color=color, lw=3, label=feat)
            for feat, color in zip(top_feats, colors)
        ]

        fig.legend(
            handles=legend_patches,
            title="Features",
            loc="center left",
            bbox_to_anchor=(1, 0.5),  # Push legend slightly to the right
            fontsize=14,
            title_fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space on right for legend

        if save_data:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(self.plot_dir / f"{filename}{exp_suffix}.png", dpi=dpi)

        plt.show()

    def plotUncertaintyEvolution(self,
                                    experiment_ls: list,
                                    n_iters: int = 30,
                                    n_bins: int = 3,
                                    generic_pred_filename: str = "all_preds_*.csv.gz", 
                                    save_plot: bool=False,
                                    tick_fontsize:int=18,
                                    label_fontsize:int=20,
                                    legend_fontsize:int=16,
                                    method_legend_map: dict=None):

        glob_best_pred = float('inf')
        glob_worst_pred = float('-inf')

        #Compute global min/max for binning
        for exp in experiment_ls:
            for it in range(n_iters + 1):
                pred_file_ls = glob(self.results_dir / exp / f"it{it}" / generic_pred_filename)
                for file in pred_file_ls:
                    working_df = pd.read_csv(file, index_col='ID')
                    if 'pred_Affinity(kcal/mol)' in working_df.columns:
                        best_pred = working_df['pred_Affinity(kcal/mol)'].min()
                        worst_pred = working_df['pred_Affinity(kcal/mol)'].max()
                        if best_pred < glob_best_pred:
                            glob_best_pred = best_pred
                        if worst_pred > glob_worst_pred:
                            glob_worst_pred = worst_pred

        self.logger.info(f"Global best pred: {glob_best_pred}")
        self.logger.info(f"Global worst pred: {glob_worst_pred}")

        bin_edges = np.linspace(glob_best_pred, glob_worst_pred, n_bins + 1)
        bin_labels = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(n_bins)]

        glob_uncert_dict = {}

        # Aggregate uncertainty values binned by predicted affinity
        for exp in experiment_ls:
            self.logger.info(f"\nAnalysing Experiment: {exp}")
            experiment_dir = self.results_dir / exp

            with open(Path(exp) / "run_params.json", "r") as f:
                param_dict = json.load(f)
            step = int(param_dict["batch_size"]) 

            exp_binned_uncert_dict = {}

            for it in range(n_iters + 1):
                self.logger.info(f"  Iteration {it}")
                it_binned_uncert_dict = {}

                pred_file_ls = glob(self.results_dir + f"/{exp}/it{it}/" + generic_pred_filename)
                for file in pred_file_ls:
                    file_number = re.findall(r'\d+', file)[0]
                    working_df = pd.read_csv(file, index_col='ID')

                    if 'pred_Affinity(kcal/mol)' not in working_df.columns or 'Uncertainty' not in working_df.columns:
                        self.logger.warning(f" Skipping file (missing column): {file}")
                        continue

                    working_df['bin'] = pd.cut(
                        working_df['pred_Affinity(kcal/mol)'],
                        bins=bin_edges,
                        include_lowest=True
                    )

                    uncertainty_bin_means = working_df.groupby('bin')['Uncertainty'].mean().tolist()
                    it_binned_uncert_dict[file_number] = uncertainty_bin_means

                uncert_bin_array = np.array(list(it_binned_uncert_dict.values()))
                uncert_bin_means_across_files = np.nanmean(uncert_bin_array, axis=0).tolist()

                if isinstance(uncert_bin_means_across_files, float):
                    uncert_bin_means_across_files = [uncert_bin_means_across_files]

                n_mols = it * step
                exp_binned_uncert_dict[n_mols] = uncert_bin_means_across_files

            glob_uncert_dict[exp] = exp_binned_uncert_dict

        self.logger.debug(glob_uncert_dict)


        # Create a single figure for all experiments
        fig, ax = plt.subplots(figsize=(14, 8))

        line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]  # Extend if needed
        seen_experiments = set()

        # Loop over experiments and plot on the same figure
        for exp in experiment_ls:
            mol_counts = sorted(glob_uncert_dict[exp].keys())
            n_bins = len(glob_uncert_dict[exp][mol_counts[0]])

            # Build trajectories
            uncert_bin_trajectories = [[] for _ in range(n_bins)]
            for n_mols in mol_counts:
                uncert_vals = glob_uncert_dict[exp][n_mols]
                for i in range(n_bins):
                    uncert_bin_trajectories[i].append(uncert_vals[i])

            # Choose color for this experiment
            plot_color = 'black'
            exp_suffix = next((key for key in self.method_colour_map if exp.endswith(key)), None)
            exp_label = exp_suffix if exp_suffix else exp.split("_")[-1]
            if exp_suffix in self.method_colour_map:
                plot_color = self.method_colour_map[exp_suffix]

            # Keep track of experiment for legend (avoid duplicates)
            seen_experiments.add(exp_label)

            # Plot each bin's line for this experiment
            for i, bin_vals in enumerate(uncert_bin_trajectories):
                linestyle = line_styles[i % len(line_styles)]
                ax.plot(
                    mol_counts, bin_vals,
                    color=plot_color,
                    linestyle=linestyle,
                    linewidth=2
                )

        # Create first legend for bin styles
        bin_lines = [
            plt.Line2D([0], [0], color="black", linestyle=line_styles[i])
            for i in range(n_bins)
        ]
        bin_labels = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" for i in range(n_bins)]

        leg1 = fig.legend(
            bin_lines,
            bin_labels,
            title="Affinity Bins",
            loc="upper left",
            bbox_to_anchor=(1.01, 0.95),
            ncol=1,
            borderaxespad=0.0,
            prop={"size": legend_fontsize}
        )

        # Detect suffixes actually used in the experiment list
        used_suffixes = set()
        for exp in experiment_ls:
            matched_suffix = next((s for s in self.method_colour_map if exp.endswith(s)), None)
            if matched_suffix:
                used_suffixes.add(matched_suffix)

        # Build color legend using method_legend_map in order
        handles = []
        labels = []

        if method_legend_map:
            for suffix, label in method_legend_map.items():
                if suffix in used_suffixes:
                    colour = self.method_colour_map.get(suffix)
                    if colour:
                        handles.append(Line2D([0], [0], color=colour, lw=2))
                        labels.append(label)
                    else:
                        self.logger.warning(f"Warning: No color found for method suffix {suffix}")

        leg2 = fig.legend(
            handles,
            [label.lstrip('_') for label in labels],
            title="Experiment",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            borderaxespad=0.0,
            prop={"size": legend_fontsize}
        )

        # Make sure both legends appear
        fig.add_artist(leg1)
        fig.add_artist(leg2)

        # Final plot formatting
        ax.set_xlabel("Number of Molecules", fontsize=label_fontsize)
        ax.set_ylabel("Mean Uncertainty", fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

        plt.subplots_adjust(right=0.75)

        # Save the plot if required
        if save_plot:
            plt.savefig(Path(self.plot_dir) / 'uncertainty_evolution.png', bbox_extra_artists=(leg1, leg2), bbox_inches='tight', dpi=500)

        plt.show()

    def analyseHits(
        self,
        experiment_ls: list,
        it_ls: list,
        docking_results_path: str,
        hit_targets:str,
        top_n: int = None,
        docking_column: str = "Affinity(kcal/mol)",
        preds_column: str = "pred_Affinity(kcal/mol)",
        percentile: float = 0.01,
        preds_file: str = "all_preds_*.csv.gz",
        json_name: str = "hit_discovery",
        draw_mols: bool = True,
        plot_bars: bool = True,
        top_n_bar: int = 50,
    ):
        """Main method to compute, draw, and plot hits."""
        
        self._computeHits(
            experiment_ls, it_ls, docking_results_path,
            top_n, docking_column, preds_column, percentile, preds_file
        )

        # Save to JSON
        json_path = self.plot_dir / f"{json_name}.json"
        with open(json_path, "w") as f:
            json.dump(self.experiment_hits, f, indent=4)

        # Optionally draw final hit molecules
        if draw_mols:
            self._drawFinalHitMolecules(experiment_ls, docking_results_path)

        # Optionally plot hit discovery bars
        if plot_bars:
            self._plotDiscoveryBars(
                experiment_hits_path=str(json_path),
                top_n=top_n_bar,
            )

    def _computeHits(
            self,
            experiment_ls: list,
            it_ls: list,
            docking_results_path: str,
            top_n: int = None,
            docking_column: str = "Affinity(kcal/mol)",
            preds_column: str = "pred_Affinity(kcal/mol)",
            percentile: float = 0.01,
            preds_file: str = "all_preds_*.csv.gz",
            json_name: str = "hit_discovery"
    ):
        self.experiment_hits = {}
        docking_results_ls = glob(str(docking_results_path))

        global_docking_df = pd.DataFrame()
        for docking_file in docking_results_ls:
            df = pd.read_csv(docking_file, index_col="ID")
            df = df.sort_values(by=docking_column)
            df = df.head(int(len(df) * (percentile * 10)))
            global_docking_df = pd.concat([global_docking_df, df])

        global_docking_df = pd.to_numeric(global_docking_df[docking_column], errors='coerce').dropna().to_frame()
        global_docking_df = global_docking_df.sort_values(by=docking_column)
        final_len = int(len(global_docking_df) * percentile)
        global_docking_df = global_docking_df.head(final_len)

        for exp in experiment_ls:
            exp_path = self.results_dir / exp
            n_its = countIterations(exp_path)

            total_hits = 0
            total_hit_ids = set()
            it_hits, it_hit_ids_all = [], []
            new_hits_per_iter, new_hits_ids_all = [], []
            rediscovered_per_iter, rediscovered_ids_all = [], []

            if not it_ls:
                it_ls = [n for n in range(n_its + 1)]

            # Initialize all accumulators
            total_hit_ids = set()
            it_hits = []
            it_hit_ids_all = []
            new_hits_per_iter = []
            new_hits_ids_all = []
            rediscovered_per_iter = []
            rediscovered_ids_all = []
            top_pred_per_iter = []

            for it in it_ls:
                it_hit, it_hit_ids = 0, []
                new_hits, new_hits_ids = 0, []
                rediscovered, rediscovered_ids = 0, []
                total_count = 0

                it_path = exp_path / f'/it{it}/'
                preds_files = glob(str(it_path / preds_file))

                top_df_ls = []
                for pred_file in preds_files:
                    try:
                        df = pd.read_csv(pred_file, index_col="ID", compression="gzip")
                        df = df.sort_values(by=preds_column, ascending=True)
                        df = df.head(final_len)
                        top_df_ls.append(df)
                    except Exception as e:
                        self.logger.warning(f"Error reading {pred_file}: {e}")
                        continue

                if not top_df_ls:
                    continue

                full_df = pd.concat(top_df_ls).sort_values(by=preds_column, ascending=True)
                if not top_n:
                    top_n = final_len
                top_mols = full_df.head(top_n)

                for mol in top_mols.index:
                    if mol in global_docking_df.index:
                        total_count += 1
                        it_hit_ids.append(mol)
                        it_hit += 1
                        if mol in total_hit_ids:
                            rediscovered += 1
                            rediscovered_ids.append(mol)
                        else:
                            new_hits += 1
                            new_hits_ids.append(mol)
                            total_hit_ids.add(mol)

                it_hits.append(it_hit)
                it_hit_ids_all.append(it_hit_ids)
                new_hits_per_iter.append(new_hits)
                new_hits_ids_all.append(new_hits_ids)
                rediscovered_per_iter.append(rediscovered)
                rediscovered_ids_all.append(rediscovered_ids)

                total_hits = len(total_hit_ids)  
                top_pred_ids = list(top_mols.head(top_n).index)
                top_pred_per_iter.append(top_pred_ids)

                              

            self.experiment_hits[exp] = {
                "total_hits": total_hits,
                "total_hit_ids": list(total_hit_ids),
                "it_hits": it_hits,
                "it_hit_ids": it_hit_ids_all,
                "new_hits": new_hits_per_iter,
                "new_hits_ids": new_hits_ids_all,
                "rediscovered": rediscovered_per_iter,
                "rediscovered_ids": rediscovered_ids_all,
                "top_pred_per_iter": top_pred_per_iter
            }

        with open(f"{PROJ_DIR}/results/plots/{json_name}.json", "w") as f:
            json.dump(self.experiment_hits, f, indent=4)

    def _drawFinalHitMolecules(
            self,
            experiment_ls: list,
            hit_targets: str,
            docking_results_path: str = PROJ_DIR / "datasets" / "PyMolGen" / "docking" / "PMG_docking_*.csv",
    ):
        with open(self.plot_dir / "hit_discovery.json", "r") as f:
            experiment_hits = json.load(f)

        self.logger.debug(f"Printing experiment hits:\n{experiment_hits}")
        hit_csv = pd.read_csv(hit_targets, index_col='ID')
        hit_csv = hit_csv.sort_values(by='Affinity(kcal/mol)').reset_index()

        for experiment in experiment_ls:
            hit_df = pd.DataFrame()

            final_it_hits = experiment_hits[experiment]["it_hit_ids"][-1]
            final_it_top_preds = experiment_hits[experiment]["top_pred_per_iter"][-1]

            hit_df['ID'] = final_it_top_preds
            is_hit_ls = []
            batch_no_ls = []
            for id in hit_df['ID']:
                if id in final_it_hits:
                    is_hit_ls.append(1)
                else:
                    is_hit_ls.append(0)
                batch_no = molid2BatchNo(
                    molid=id, 
                    prefix='PMG-',
                    dataset_file=docking_results_path
                                         )
                batch_no_ls.append(batch_no)

            hit_df['is_hit'] = is_hit_ls
            hit_df['batch_no'] = batch_no_ls

            hit_df = hit_df.sort_values(by='batch_no')

            smiles_df_ls = []
            for batch_no in hit_df['batch_no'].unique():
                batch_path = docking_results_path.replace("*", str(batch_no))
                batch_df = pd.read_csv(batch_path, index_col='ID')
                
                molid_ls = []
                smiles_ls = []
                for id in hit_df[hit_df['batch_no'] == batch_no]['ID']:
                    smiles_ls.append(batch_df.loc[id, 'SMILES'])
                    molid_ls.append(id)

                smiles_df = pd.DataFrame(index=molid_ls)
                smiles_df['SMILES'] = smiles_ls
                smiles_df_ls.append(smiles_df)
            
            top_pred_smiles_df = pd.concat(smiles_df_ls)

            self.logger.debug(hit_df.head(1))
            self.logger.debug(top_pred_smiles_df.head(1))

            # Make sure both use the same index (ID)
            hit_df = hit_df.set_index("ID")
            top_pred_smiles_df.index.name = "ID"

            # Now join works as expected
            hit_df = hit_df.join(top_pred_smiles_df, how='left')

            # Convert SMILES to RDKit Mol objects
            hit_df["mol"] = hit_df["SMILES"].apply(Chem.MolFromSmiles)
            hit_df = hit_df[hit_df["mol"].notnull()]  # Filter out failed parses

            # Generate legends based on 'is_hit' column
            hit_df["legend"] = hit_df.apply(lambda row: f"{row.name} - {'HIT' if row['is_hit'] else ''}", axis=1)

            # Draw the molecules
            img = Draw.MolsToGridImage(
                hit_df["mol"].tolist(),
                legends=hit_df["legend"].tolist(),
                molsPerRow=5,
                subImgSize=(200, 200),
            )

            img_pil = PILImage.open(BytesIO(img.data))
            img_pil.save(self.plot_dir / f"{experiment}_final_hits_grid.png")



    def _plotDiscoveryBars(self,
                            experiment_hits_path: str = f"{RESULTS_DIR}/plots/hit_discovery.json",
                            label_fontsize: int = 22,
                            legend_fontsize: int = 18,
                            tick_fontsize: int = 18,
                            top_n: int = 50,
                            plot_name: str = "hits_discovery",
                            allowed_experiments: list = [],
                            method_legend_map: dict = None,
                            n_its:int=30):
        
        if experiment_hits_path:
            with open(experiment_hits_path, "r") as f:
                self.experiment_hits = json.load(f)

        experiments = (
            [key for key in self.experiment_hits if any(key.endswith(ae) for ae in allowed_experiments)]
            if allowed_experiments
            else list(self.experiment_hits.keys())
        )

        n_exps = len(experiments)

        if not n_its:
            n_its = max(len(data["new_hits"]) for data in self.experiment_hits.values())

        iterations = np.arange(n_its)
        bar_width = 0.8 / n_exps

        fig, ax1 = plt.subplots(figsize=(18, 9))
        ax2 = ax1.twinx()

        colour_ls = []
        linestyle_ls = []
        exp_suffixes = []

        # Build colors and linestyles
        for exp in experiments:
            exp_suffix = next((s for s in self.method_colour_map if exp.endswith(s)), None)
            exp_suffixes.append(exp_suffix if exp_suffix else "_unknown")
            colour_ls.append(self.method_colour_map.get(exp_suffix, "black"))
            linestyle_ls.append(self.linestyles.get("_50_" if "_50_" in exp else "_10_", "--"))

        method_color_legend = {}

        for i, (exp, exp_suffix) in enumerate(zip(experiments, exp_suffixes)):
            new_hits = self.experiment_hits[exp]["new_hits"][:n_its]
            rediscovered = self.experiment_hits[exp]["rediscovered"][:n_its]

            # If too short, pad to n_its
            new_hits += [0] * (n_its - len(new_hits))
            rediscovered += [0] * (n_its - len(rediscovered))

            total_counts = [n + r for n, r in zip(new_hits, rediscovered)]

            color = colour_ls[i]
            linestyle = linestyle_ls[i]
            x_positions = iterations + i * bar_width

            ax1.bar(x_positions, rediscovered, width=bar_width, color=color, alpha=0.25, hatch='///')
            ax1.bar(x_positions, new_hits, width=bar_width, bottom=rediscovered, color=color, alpha=0.6)

            method_color_legend[exp_suffix] = color

            discovery_pct = [(oc / top_n) * 100 for oc in total_counts]
            ax2.plot(x_positions, discovery_pct, marker='o', linestyle=linestyle, color=color,
                    linewidth=4, markersize=8)

        ax1.set_xlabel("Iteration", fontsize=label_fontsize)
        ax1.set_ylim(0, 50)
        ax1.set_ylabel("Hit Count", fontsize=label_fontsize)
        ax2.set_ylabel("Hit Discovery %", fontsize=label_fontsize)
        ax2.set_ylim(0, 100)

        ax1.set_xticks(iterations + bar_width * (n_exps - 1) / 2)
        ax1.set_xticklabels([str(i) for i in iterations], rotation=60, fontsize=tick_fontsize)
        ax1.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)

        # Plot Elements Legend (left of plot)
        element_handles = [
            Patch(facecolor='gray', alpha=0.4, label='New Hits'),
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Rediscovered Hits'),
            Line2D([0], [0], color='black', linestyle='-', marker='o', linewidth=2, label='Hit Discovery %')
        ]
        leg1 = fig.legend(
            handles=element_handles,
            title="Plot Elements",
            loc='center left',
            bbox_to_anchor=(0.75, 0.85),
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize
        )

        method_handles = []
        method_labels = []

        # Only include suffixes present in this plot
        suffixes_in_plot = set(exp_suffixes)

        if method_legend_map:
            for suffix, label in method_legend_map.items():
                if suffix in suffixes_in_plot:
                    color = self.method_colour_map.get(suffix)
                    if color:
                        method_handles.append(Line2D([0], [0], color=color, lw=4))
                        method_labels.append(label)

        leg2 = fig.legend(
            method_handles,
            method_labels,
            title="Methods",
            loc='center left',
            bbox_to_anchor=(0.75, 0.5),
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize
        )

        # Final adjustments
        plt.subplots_adjust(right=0.75)
        fig.tight_layout(rect=[0, 0, 0.75, 1])
        fig.add_artist(leg1)
        fig.add_artist(leg2)

        plt.savefig(self.plot_dir / f"{plot_name}.png")
        plt.show()

    def uncertaintyDevelopment(
            self,
            experiment: str,
            true_values_path: str,
            final_it: int = 30,
            bins: list = [0.0, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels: list = [
                "Very Certain",
                "Moderately Certain",
                "Certain",
                "Quite Uncertain",
                "Very Uncertain"
            ],
            tick_fontsize:int=18,
            label_fontsize:int=20,
            plot_name: str="uncertainty_rmse_bin_plot",
            save_plot:bool=True,
    ):
        
        full_dict = {}

        # Load true values once
        true_df = pd.read_csv(true_values_path, index_col="ID")
        true_df['Affinity(kcal/mol)'] = pd.to_numeric(true_df['Affinity(kcal/mol)'], errors='coerce')

        for it in range(0, final_it + 1):
            bin_rmse_list = {label: [] for label in labels}
            bin_counts_total = {label: 0 for label in labels}

            preds_path = f"{self.results_dir}/{experiment}/it{it}/held_out_preds.csv"

            try:
                df = pd.read_csv(preds_path, index_col="ID")
            except FileNotFoundError:
                self.logger.error(f"File not found for iteration {it}: {preds_path}")
                continue

            # Ensure numeric
            df['Uncertainty'] = pd.to_numeric(df['Uncertainty'], errors='coerce')
            df['pred_Affinity(kcal/mol)'] = pd.to_numeric(df['pred_Affinity(kcal/mol)'], errors='coerce')

            # Merge true values
            df = df.join(true_df[["Affinity(kcal/mol)"]], how="inner")

            # Drop NaNs
            df = df.dropna(subset=["Uncertainty", "pred_Affinity(kcal/mol)", "Affinity(kcal/mol)"])

            # Bin uncertainty
            df["uncertainty_bin"] = pd.cut(df["Uncertainty"], bins=bins, labels=labels)

            # Compute per-molecule RMSE
            df["RMSE"] = np.sqrt((df["Affinity(kcal/mol)"] - df["pred_Affinity(kcal/mol)"]) ** 2)

            # Count and compute per-bin RMSE
            mean_rmse_per_bin = {}
            bin_counts = df['uncertainty_bin'].value_counts().reindex(labels, fill_value=0).to_dict()

            for label in labels:
                bin_df = df[df["uncertainty_bin"] == label]
                bin_counts_total[label] += bin_counts[label]
                
                if not bin_df.empty:
                    rmse = np.sqrt(
                        mean_squared_error(
                            bin_df["Affinity(kcal/mol)"],
                            bin_df["pred_Affinity(kcal/mol)"]
                        )
                    )
                    mean_rmse_per_bin[label] = round(rmse, 2)
                else:
                    mean_rmse_per_bin[label] = np.nan

            full_dict[it] = {
                "bin_counts": bin_counts_total,
                "mean_rmse_per_bin": mean_rmse_per_bin
            }
        
        data = {label: [] for label in labels}
        iterations = sorted(full_dict.keys())

        for it in iterations:
            for label in labels:
                rmse = full_dict[it]["mean_rmse_per_bin"].get(label, None)
                data[label].append(rmse)

        # Create DataFrame
        df_rmse = pd.DataFrame(data, index=iterations)
        df_rmse.index.name = "Iteration"

        label_colors = {
            "Very Certain": "#2ca02c",        # green
            "Moderately Certain": "#98df8a",  # light green
            "Certain": "#ffcc00",             # yellow
            "Quite Uncertain": "#ff7f0e",     # orange
            "Very Uncertain": "#d62728",      # red
        }

        plt.figure(figsize=(10, 6))
        for label in labels:
            plt.plot(
                df_rmse.index,
                df_rmse[label],
                marker="o",
                label=label,
                color=label_colors[label]
            )

        plt.xlabel("Iteration", fontsize=label_fontsize)
        plt.ylabel("Mean RMSE", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)

        plt.legend(
            title="Uncertainty Bin",
            fontsize=tick_fontsize,
            title_fontsize=tick_fontsize,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )

        plt.tight_layout()

        count_table = pd.DataFrame({
            it: full_dict[it]["bin_counts"] for it in iterations
        }).T  # Transpose to have iterations as rows

        count_table.index.name = "Iteration"

        # Convert to percentages (row-wise normalization)
        total_mols = len(true_df)  # total number of molecules
        count_table_pct = (count_table / total_mols) * 100        
        count_table_pct = count_table_pct.round(1)  # Round to 1 decimal place

        combined_columns = pd.MultiIndex.from_product(
            [count_table.columns, ["Count", "%"]],
            names=['Bin', "Metric"]
        )

        combined_data = []

        for it in count_table.index:
            row= []
            for col in count_table.columns:
                row.extend([count_table.loc[it,col], count_table_pct.loc[it,col]])
            
            combined_data.append(row)
        combined_df = pd.DataFrame(combined_data, index=count_table.index, columns=combined_columns)
        combined_df.index.name = "Iteration"

        if save_plot:
            plt.savefig(self.plot_dir / f"{plot_name}.png")
            combined_df.to_csv(self.plot_dir / f"{plot_name}.csv")
        plt.show()

        return count_table_pct, full_dict
