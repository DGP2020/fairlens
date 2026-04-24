from fastapi import FastAPI, HTTPException
import pandas as pd
from .proxy_detector import ProxyDetector
from .remediation import SyntheticRepairEngine
from .audit import AuditEngine

app = FastAPI(title="FairLens API")

@app.post("/audit")
async def run_audit(data: list, target: str, protected_col: str, unprivileged_val: str, privileged_val: str):
    try:
        df = pd.DataFrame(data)
        auditor = AuditEngine()
        return auditor.calculate_dir(df, target, protected_col, unprivileged_val, privileged_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/proxy-check")
async def proxy_check(data: list, protected_col: str):
    try:
        df = pd.DataFrame(data)
        detector = ProxyDetector()
        return detector.detect_proxies(df, protected_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/repair")
async def repair_data(data: list, target: str, protected_col: str, cat_cols: list):
    try:
        df = pd.DataFrame(data)
        engine = SyntheticRepairEngine(protected_col, cat_cols)
        repaired_df = engine.generate_fair_data(df, target)
        return repaired_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))