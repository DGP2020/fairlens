import pandas as pd

class AuditEngine:
    def calculate_dir(self, df: pd.DataFrame, target: str, protected_col: str, unprivileged_val, privileged_val, positive_val=None):
        # Filter for unprivileged and privileged groups
        unprivileged_df = df[df[protected_col] == unprivileged_val]
        privileged_df = df[df[protected_col] == privileged_val]
        
        if len(unprivileged_df) == 0 or len(privileged_df) == 0:
            return {"disparate_impact_ratio": 0.0, "status": "ERROR", "message": "One or both groups have no data."}

        # Calculate probability of positive outcome
        if positive_val is not None:
            # If a specific positive value is provided
            p_unprivileged = (unprivileged_df[target] == positive_val).mean()
            p_privileged = (privileged_df[target] == positive_val).mean()
        else:
            # Assume target is numeric and higher is better
            try:
                p_unprivileged = unprivileged_df[target].mean()
                p_privileged = privileged_df[target].mean()
            except Exception:
                # Fallback: find the most frequent value in the whole dataset as positive
                most_frequent = df[target].mode()[0]
                p_unprivileged = (unprivileged_df[target] == most_frequent).mean()
                p_privileged = (privileged_df[target] == most_frequent).mean()
        
        dir_score = p_unprivileged / p_privileged if p_privileged > 0 else 1.0
        
        # Fair range is typically 0.8 to 1.25
        status = "FAIL" if dir_score < 0.8 or dir_score > 1.25 else "PASS"
        return {"disparate_impact_ratio": round(dir_score, 4), "status": status}