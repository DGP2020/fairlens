import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

class ProxyDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect_proxies(self, df: pd.DataFrame, protected_col: str):
        df_encoded = df.apply(LabelEncoder().fit_transform)
        X = df_encoded.drop(columns=[protected_col])
        y = df_encoded[protected_col]
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_results = dict(zip(X.columns, mi_scores))
        proxies = {f: s for f, s in mi_results.items() if s > self.threshold}
        return {"scores": mi_results, "proxies": proxies}