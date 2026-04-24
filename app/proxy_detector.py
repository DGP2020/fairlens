import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

class ProxyDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect_proxies(self, df: pd.DataFrame, protected_col: str):
        # Only encode categorical columns, preserve numeric ones
        df_encoded = df.copy()
        le = LabelEncoder()
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        X = df_encoded.drop(columns=[protected_col])
        y = df_encoded[protected_col]
        
        # Identify discrete features for mutual_info_classif
        discrete_features = [df[col].dtype == 'object' or df[col].dtype.name == 'category' for col in X.columns]
        
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
        mi_results = {f: float(s) for f, s in zip(X.columns, mi_scores)}
        proxies = {f: s for f, s in mi_results.items() if s > self.threshold}
        
        return {"scores": mi_results, "proxies": proxies}