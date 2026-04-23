import pandas as pd

class AuditEngine:
    def calculate_dir(self, df: pd.DataFrame, target: str, protected_col: str, unprivileged_val, privileged_val):
        # Probability of positive outcome for unprivileged group
        unprivileged_outcomes = df[df[protected_col] == unprivileged_val][target]
        p_unprivileged = unprivileged_outcomes.mean() if len(unprivileged_outcomes) > 0 else 0
        
        # Probability of positive outcome for privileged group
        privileged_outcomes = df[df[protected_col] == privileged_val][target]
        p_privileged = privileged_outcomes.mean() if len(privileged_outcomes) > 0 else 0
        
        dir_score = p_unprivileged / p_privileged if p_privileged > 0 else 1.0
        
        status = "FAIL" if dir_score < 0.8 or dir_score > 1.25 else "PASS"
        return {"disparate_impact_ratio": round(dir_score, 4), "status": status}