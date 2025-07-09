# SimDMTA Workflow: Simulation Framefork for Iterative Docking-Based Molecular Selection Assessment

## Project Overview

The SimDMTA project aims to accelerate the early stages of drug discovery by *simulating* the pharmaceutical Design–Make–Test–Analyse (*DMTA*) cycle. 
This simulation enables us to evaluate the downstream effects of various molecule selection strategies on machine learning model performance.
As AI becomes increasingly integrated into pharmaceutical research, it is essential to recognise the impact of model retraining on the quality and efficiency of candidate selection. 
SimDMTA provides a framework for systematically assessing how selection strategies influence predictive performance and a model's hit-finding capabilities.

## Accompanying Paper
In progress

![image](https://github.com/user-attachments/assets/aaa0d982-6aef-4bd9-b85c-f61f54285ee7)


## Molecule Selection/ Query Strategies

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

## Directory structure

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
    │   │   ├── training_init_model.py
    │   │   └── Submit_init_train.sh
    │   ├── Submit_calc_new_test.sh
    │   └── RF_class.py                   # Contains RFModel class
    │
    ├── mol_sel/                          
    │   └── mol_sel_fns.py                # Molecule selection logic
    │
    └── run/
        ├── Submit_run_DMTA.sh
        ├── run_DMTA.py                   # Main run script
        ├── workflow_fns.py            
        ├── Submit_analysis.sh                    
        ├── analysis_fns.py                     
        ├── run_analysis.py                      
        ├── Submit_average_all.sh                      
        ├── average_fns.py                    
        └── run_average.py
```

## References
GNINA: 

https://github.com/gnina/gnina

Lilly-MedChem-Rules:

https://github.com/IanAWatson/Lilly-Medchem-Rules
