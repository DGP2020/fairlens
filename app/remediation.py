import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC, SMOTE

class SyntheticRepairEngine:
    def __init__(self, protected_col: str, cat_cols: list):
        self.protected_col = protected_col
        self.cat_cols = cat_cols

    def generate_fair_data(self, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])
        y = df[target]
        
        # Create a composite label to balance across both target and protected class
        composite_y = y.astype(str) + "_" + df[self.protected_col].astype(str)
        
        # Identify categorical indices
        cat_idx = [X.columns.get_loc(c) for c in self.cat_cols if c in X.columns]
        
        if cat_idx:
            smote = SMOTENC(categorical_features=cat_idx, random_state=42)
        else:
            # Fallback to standard SMOTE if no categorical features are present
            smote = SMOTE(random_state=42)
            
        X_syn, y_comp_syn = smote.fit_resample(X, composite_y)
        
        df_syn = pd.DataFrame(X_syn, columns=X.columns)
        
        # Restore the original target from the composite labels
        df_syn[target] = y_comp_syn.str.split("_").str[0].astype(y.dtype)
        return df_syn