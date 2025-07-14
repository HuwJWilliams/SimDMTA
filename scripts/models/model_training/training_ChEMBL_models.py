import sys
from pathlib import Path
from glob import glob
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / 'scripts'
RESULTS_DIR = PROJ_DIR / 'results'
DATASET_DIR = PROJ_DIR / "datasets"

print(FILE_DIR)
print(PROJ_DIR)
print(SCRIPTS_DIR)
print(RESULTS_DIR)
print(DATASET_DIR)

sys.path.insert(0, str(FILE_DIR.parent))
from RF_class import RFModel
import pandas as pd

sys.path.insert(0, SCRIPTS_DIR / "misc")
from misc_functions import readConfigJSON

n_resamples = 50
inner_cv_type = ("kfold",)
n_splits = 5
tr_te_split = 0.3
search_type = "grid"
loss_function = "neg_mean_squared_error"
docking_column = "Affinity(kcal/mol)"

hyper_params = {
    "rf__n_estimators": [400, 500],
    "rf__max_features": ["sqrt"],
    "rf__max_depth": [10, 20, 30, 50],
    "rf__min_samples_split": [10, 20, 30],
    "rf__min_samples_leaf": [10, 20, 30],
}

config_json = readConfigJSON(config_fpath=PROJ_DIR / 'config.json')
model_path = config_json['it0_model_dir']
data_paths = config_json['data']

targs_df = pd.read_csv(data_paths["it0_training_dock"], index_col="ID")

# Setting up input targets, removing any ones which failed to dock
targs = targs_df[["Affinity(kcal/mol)"]]
falsetargs = targs_df[targs_df["Affinity(kcal/mol)"] == "False"]
targs = targs.drop(index=falsetargs.index)

feats_df = pd.read_csv(
    data_paths["it0_training_desc"],
    index_col="ID",
)
save_path = Path(model_path)

save_path.mkdir(exist_ok=True)
training_data_path = save_path / "training_data"
training_data_path.mkdir(exist_ok=True)

trained_features = feats_df.drop(index=falsetargs.index)

print("1: trained features")
print(trained_features)

model = RFModel(docking_column=docking_column)

model.trainRegressor(
    search_type=search_type,
    hyper_params=hyper_params,
    features=trained_features,
    targets=pd.DataFrame(targs),
    save_path=save_path,
    save_final_model=True,
    plot_feat_importance=True,
)

# Define file paths and prefixes
desc_file = data_paths["selection_pool_desc"]
full_file = data_paths["selection_pool_full"]
held_out_desc_file = data_paths["held_out_desc"]
held_out_targ_file = data_paths["held_out_dock"]


desc_df = pd.read_csv(desc_file, index_col="ID")
desc_df = desc_df.loc[:, desc_df.columns.isin(trained_features.columns)]
desc_df = desc_df.reindex(columns=trained_features.columns)
feats = desc_df

print("2: prediction features")
print(feats)

preds = model.predict(
    feats=pd.DataFrame(feats),
    save_preds=True,
    calc_mpo=True,
    full_data_fpath=full_file,
    preds_filename=f"all_preds_{1}",
    preds_save_path=save_path,
)

print("3")
print(preds)

# Predicting on held out

rf_pkl = save_path / "final_model.pkl"
rf_model = joblib.load(rf_pkl)

with open(rf_pkl, "rb") as feats:
    data = pickle.load(feats)

feats_df = pd.read_csv(held_out_desc_file, index_col="ID")
ho_df = pd.read_csv(held_out_targ_file, index_col="ID")

ho = ho_df[[docking_column]]
falseho = ho_df[ho_df[docking_column] == "False"]
ho_ = ho.drop(index=falseho.index)

same_value_columns = feats_df.columns[feats_df.nunique() > 1]
new_feat_df = feats_df[same_value_columns]
new_feat_df = new_feat_df.drop(index=falseho.index)

# Convert 'data' to a list of expected feature names
expected_features = data if isinstance(data, list) else data.tolist()

# Ensure all expected features exist in new_feat_df
for col in expected_features:
    if col not in new_feat_df.columns:
        new_feat_df[col] = 0  # or np.nan if appropriate

# Reorder to match training feature order
new_feat_df = new_feat_df[expected_features]

# Reindex to ensure ID order/labels match original descriptor file
new_feat_df = new_feat_df.reindex(feats_df.index)

# Now predict using the reordered features
preds = rf_model.predict(new_feat_df)

# Build predictions DataFrame using original index
preds_df = pd.DataFrame(
    {"pred_Affinity(kcal/mol)": preds},
    index=new_feat_df.index
)
preds_df.to_csv(save_path / "held_out_preds.csv", index_label="ID")

ho_[f"pred_{docking_column}"] = preds

true = ho_[docking_column].astype(float)
pred = ho_[f"pred_{docking_column}"].astype(float)

errors = true - pred

# Calculate performance metrics
bias = np.mean(errors)
sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
mse = mean_squared_error(true, pred)
rmse = np.sqrt(mse)
r2 = r2_score(true, pred)

metrics = {
    "Bias": round(bias, 3),
    "SDEP": round(sdep, 3),
    "MSE": round(mse, 3),
    "RMSE": round(rmse, 3),
    "r2": round(r2, 3),
}

import json

with open(save_path / "held_out_stats.json", "w") as file:
    json.dump(metrics, file, indent=4)
