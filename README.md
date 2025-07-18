# SimDMTA Workflow

![image](https://github.com/user-attachments/assets/aaa0d982-6aef-4bd9-b85c-f61f54285ee7)

## 🧠 Project Overview

The SimDMTA project aims to accelerate the early stages of drug discovery by *simulating* the pharmaceutical Design–Make–Test–Analyse (*DMTA*) cycle. 
This simulation enables us to evaluate the downstream effects of various molecule selection strategies on machine learning model performance.
As AI becomes increasingly integrated into pharmaceutical research, it is essential to recognise the impact of model retraining on the quality and efficiency of candidate selection. 
SimDMTA provides a framework for systematically assessing how selection strategies influence predictive performance and a model's hit-finding capabilities.

## 📖 Citation
If you use this framework in your research, please cite:
[Manuscript in progress]

## 🔬 Molecule Selection/ Query Strategies

### MP (Most Potent)
Selects molecules with the best predicted docking scores (i.e., most negative binding affinities).
Type: Exploitative / Greedy

### MPO (Multi-Parameter Optimisation)

Selects molecules with the best MPO (Multi-Parameter Optimization) score, integrating predicted activity with physicochemical properties (e.g., LogP, PFI).
Type: Exploitative / Greedy

### MU (Most Uncertain)

Selects molecules with the highest prediction uncertainty, based on variance across the ensemble of random forest trees.
Type: Explorative

### R (Random)

Selects molecules entirely at random from the pool.
Type: Baseline / Neutral

### RMP (Random in Most Potent)

Selects randomly from the top X% (e.g., 10%) of molecules ranked by predicted docking score.
Type: Balanced (Exploitative with some exploration)

### RMPO (Random in Most MPO)

Selects randomly from the top X% (e.g., 10%) of molecules ranked by MPO score.
Type: Balanced (Exploitative with some exploration)

### RMU (Random in Most Uncertain)

Selects randomly from the top X% of molecules with the highest prediction uncertainty.
Type: Balanced / Explorative

### HYBRID (Custom combinations like rmp:mu)

Combines multiple strategies in series (e.g., first filter by RMP, then pick the most uncertain).
Type: User-defined mix of exploration and exploitation

## 📁 Directory structure

```text
project/
├── datasets/
│
├── docking/
│
├── results/
│   ├── init_RF_model/                    # Initial trained RF model
│   └── plots/                            # Directory to keep plots
│
└── scripts/
    ├── dataset/                          # Dataset utilities
    │   ├── Lilly-Medchem-Rules           # Can be installed via GitHub (See reference section)
    │   └── dataset_fns.py                # Contains dataset classes
    │
    ├── docking/                          
    │   ├── receptors/                    # Receptor files for docking
    │   ├── docking_fns.py                # Docking functions using GNINA
    │   ├── gnina                         # GNINA installation (See reference section)
    │   └── dock_init.py                  # Docking initialization
    │
    ├── misc/                             
    │   └── misc_fns.py                   # Miscellaneous utility functions
    │
    ├── models/                           
    │   ├── model_training/
    │   │   └── training_init_model.py
    │   └── RF_class.py                   # Contains RFModel class
    │
    ├── mol_sel/                          
    │   └── mol_sel_fns.py                # Molecule selection logic
    │
    └── run/
        ├── run_DMTA.py                   # Main run script
        ├── workflow_fns.py            
        ├── analysis_fns.py                     
        ├── run_analysis.py                      
        ├── average_fns.py                    
        └── run_average.py
```
## 📦 Creating the conda environment

conda create -n simdmta_env python=3.9
conda activate simdmta_env
pip install -r requirements.txt

## 🚀 Example Usage
To run the SimDMTA workflow:

python run_DMTA.py <n_cmpds> <sel_method> <start_iter> <total_iters> <run_date> <random_frac>

### 🔧 Parameters
| Argument      | Description                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `n_cmpds`     | Number of compounds to select in each iteration.                                                                                |
| `sel_method`  | Molecule selection method. One of: `mp`, `mu`, `mpo`, `rmpo`, `rmp`, `rmu`, `r`.                                                |
| `start_iter`  | Iteration number to start from. Use `1` for new runs; higher values to resume failed ones.                                      |
| `total_iters` | Total number of iterations to run.                                                                                              |
| `run_date`    | Identifier for the run, typically the current date in `YYYYMMDD` format.                                                        |
| `random_frac` | Fraction to sample from during selection in `rmp`, `rmpo`, and `rmu` modes (e.g., `0.1` for 10%). Must be set even if not used. |

To verify installation, you can run a small test:

## 📈 Outputs
```
results/
└── <run_name>/ # Unique run directory (based on selection method, date, etc.)
├── it1/
│ ├── all_preds_1.csv.gz                 # Predictions for molecules in batch 1
│ ├── ...
│ ├── all_preds_X.csv.gz                 # Predictions for molecules in final batch
│ ├── best_params.json                   # Best hyperparameters found during training
│ ├── feature_importance.df              # Feature importance values
│ ├── feature_importance_plot.png
│ ├── final_model.pkl                    # Trained random forest model
│ ├── performance_stats.json             # Training stats (R², RMSE, etc.)
│ ├── held_out_stats.json                # Performance on held-out test set
│ └── training_data/
│ ├── training_features.csv.gz
│ └── training_targets.csv.gz
│
├── it2/
├── ...
└── itX/                                # Final iteration
```


## 📫 Contact
For questions, feedback, or collaboration ideas, feel free to reach out to:
- Huw Williams (huwjwilliams@btinternet.com)

## 🔗 References
GNINA: 

https://github.com/gnina/gnina

Lilly-MedChem-Rules:

https://github.com/IanAWatson/Lilly-Medchem-Rules
