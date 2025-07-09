# %%
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Optional
import pandas as pd
import sys
sys.path.insert(0, '/users/yhb18174/Recreating_DMTA/scripts/models')
from RF_class import RF_model
rf_model = RF_model(docking_column = 'Affinity(kcal/mol)')

# %%
class ReliabilityFilter:
    

    def __init__(
            self,
            n_estimators: int=100,
            test_size_fraction: float=0.3,
            reliability_threshold: float = 0.5,
            random_state: int = 1
    ):
        
        self.n_estimators = n_estimators
        self.test_size_fraction = test_size_fraction
        self.reliability_threshold = reliability_threshold
        self.random_state = random_state
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def calculate_reliability_scores(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 5
    ) -> np.ndarray:
        
        n_samples = len(X)
        preds = np.zeros((n_samples, self.n_estimators))

        for i, tree in enumerate(self.rf_model.estimators_):
            preds[:, i] = tree.predict(X)

        mean_preds = np.mean(preds, axis=1)
        std_preds = np.std(preds, axis=1)

        pred_error = np.abs(mean_preds - y)
        max_error = np.max(pred_error)
        normalised_error = pred_error / max_error if max_error > 0 else pred_error

        reliability_scores = 1 - (normalised_error * std_preds)

        return reliability_scores
    
# %%
