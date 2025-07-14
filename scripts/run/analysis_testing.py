
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
RESULTS_DIR = PROJ_DIR / "results"
TEST_DIR = PROJ_DIR / "datasets" / "test_dataset"
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}
sys.path.insert(0, PROJ_DIR / "scripts/run/")
from analysis_class import Analysis

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import readConfigJSON
config_json = readConfigJSON(config_fpath=PROJ_DIR / 'config.json')
model_path = config_json['it0_model_dir']
receptor_path = config_json["receptor"]


an = Analysis(results_dir=RESULTS_DIR, 
              held_out_stat_json="held_out_stats.json",
              docking_column='Affinity(kcal/mol)'
              )

chembl_feats = TEST_DIR / "ChEMBL_rdkit_desc_trimmed.csv"
ho_feats = TEST_DIR / "PMG_held_out_desc_trimmed.csv"
ho_targs = TEST_DIR / "PMG_held_out_targ_trimmed.csv"
prediction = TEST_DIR / "PMG_rdkit_desc_1.csv"
pred_full = TEST_DIR / "PMG_rdkit_1.csv.gz"
pred_dock = TEST_DIR / "PMG_docking_1.csv"
hits_feats = TEST_DIR / "hits_features.csv"
hits_targs = TEST_DIR / "hits_targets.csv"

experiment_ls = ['']

an.plotPCA(train=chembl_feats,
            validation=ho_feats,
            prediction=prediction,
            source_ls=['ChEMBL', 
                       'Held_Out', 
                       'PyMolGen'],
            n_components=5,
            plot_scatter=True,
            plot_area=True,
            save_extra_data=True,
            plot_loadings=True,
            kdep_sample_size=0.5,
            kdep_sample_ls=['PyMolGen'],
            tick_fontsize=18,
            label_fontsize=20,
            axis_fontsize=18,
            legend_fontsize=20,
           kde_tick_dicts = [
    {"xticks": [-20, 0, 20], "yticks": [0.00, 0.035, 0.07]},  # for PC1
    {"xticks": [-20, 0, 20], "yticks": [0.00, 0.05, 0.10]}, # for PC2
    {"xticks": [-10, 0, 10], "yticks": [0.00, 0.04, 0.08, 0.12]},  # for PC3
    {"xticks": [-10, 0, 10], "yticks": [0.00, 0.10, 0.20]},  # for PC4
    {"xticks": [-10, 0, 10, 20], "yticks": [0.00, 0.05, 0.1, 0.15]},  # for PC5
])

an.predictionDevelopment(experiment=experiment_ls[0],
                        prediction_fpath = "held_out_preds.csv",
                        true_path= ho_targs,
                        iter_ls=[0, 2],
                        save_plot=True,
                        it0_dir=model_path,
                        underlay_it0=True,
                        title_fontsize=24,
                        tick_fontsize=20,
                        label_fontsize=22,
                        metric_fontsize=16,
                        legend_fontsize=22,
                        x_ticks=(-10, -9, -8, -7),
                        y_ticks=(-10, -9, -8, -7),
                        br_box_position=(0.95, -0.1),
                        tl_box_position=(0.6, 0.95),
                        figsize=(15,15),
                        regression_line_colour='coral'
                        )

an.plotModelPerformance(
    experiments=experiment_ls,
    plot_ho =True,
    method_legend_map= {
                        "_mp": "MP",
                        "_mpo": "MPO",
                        "_mu": "MU",
                        "_r" : "R",
                        "_rmp": "RMP",
                        "_rmpo": "RMPO",
                        "_rmu": "RMU",
                      },
    plot_int= False,
    plot_chembl_int= False,
    plot_fname='test_plot',
    set_ylims=True,
    r_type = 'pearson_r',
    rmse_ylim=(0.25, 1.25),
    sdep_ylim=(0.0, 0.6),
    r2_ylim=(-0.2, 1),
    bias_ylim= (-1.2, 0),
    yticks=4,
    custom_xticks=[0, 500, 1000, 1500],
    tick_fontsize=20,
    label_fontsize=24,
    title_fontsize=20,
    legend_fontsize=20,
    save_plot=True,
    )



an.topPredictionAnalysis(
    experiments=experiment_ls,
    sort_by_descending=False,
    filename="Avg_Bottom_Preds_Plot",

)

an.drawChosenMols(experiment = experiment_ls[0], iter_ls = [1, 2],
                    save_img = True, img_fname='chosen_mol_img',
                    full_data_fpath=TEST_DIR / "PMG_rdkit_1.csv.gz")

an.plotPCAAcrossIters(experiment=experiment_ls[0], 
                         iter_ls=[0, 1, 2],
                         save_plot=True,
                         prediction_descs=TEST_DIR / 'PMG_rdkit_desc_1.csv',
                         chembl_descs= TEST_DIR / "ChEMBL_rdkit_desc_trimmed.csv",
                         n_rows=4,
                         plot_fname="PCA_across_iters",
                         n_components=5,
                         indi_plot_suffix='RM_outliers',
                         remove_outliers=True,
                         use_multiprocessing=True
)


%%
an.plotTopPredDocked(experiment_ls=experiment_ls,
                        iter=2,
                        data_fpath=pred_dock,
                        docking_dir=TEST_DIR,
                        receptor_path=receptor_path,
                        max_confs=100,
                        save_plot=True,
                        save_structures=True,
                        search_in_top=50,
                        plot_name="sing_50_pred_docked_boxplot",
                    )

an.uncertaintyChecker(experiment_ls=experiment_ls,
                                     iter_ls=[1, 2],
                                     save_plot=True,
                                     plot_name='uncertainty_checker',
                                     true_path=ho_targs)


an.plotFeatImportanceAndRidgelines(experiment=experiment_ls[0],
                               iter_ls=[0, 2],
                               extra_sources={
                                   "ChEMBL": chembl_feats,
                                   "Hits": hits_feats,
                               }
                               )


an.analyseHits(experiment_ls=experiment_ls,
               it_ls=[0, 2],
               docking_results_path=pred_dock,
               hit_targets=hits_targs,
               top_n=50)

an.uncertaintyDevelopment(experiment=experiment_ls[0],
                            plot_name=f"uncertainty_rmse_bin",
                            final_it=2, 
                            true_values_path=ho_targs)

