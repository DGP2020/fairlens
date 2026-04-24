from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from .proxy_detector import ProxyDetector
from .remediation import SyntheticRepairEngine
from .audit import AuditEngine

app = FastAPI(title="FairLens API")

class AuditRequest(BaseModel):
    data: List[dict]
    target: str
    protected_col: str
    unprivileged_val: str
    privileged_val: str
    positive_val: Optional[str] = None

class ProxyRequest(BaseModel):
    data: List[dict]
    protected_col: str

class RepairRequest(BaseModel):
    data: List[dict]
    target: str
    protected_col: str
    cat_cols: List[str]

@app.post("/audit")
async def run_audit(req: AuditRequest):
    try:
        df = pd.DataFrame(req.data)
        auditor = AuditEngine()
        return auditor.calculate_dir(df, req.target, req.protected_col, req.unprivileged_val, req.privileged_val, req.positive_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/proxy-check")
async def proxy_check(req: ProxyRequest):
    try:
        df = pd.DataFrame(req.data)
        detector = ProxyDetector()
        return detector.detect_proxies(df, req.protected_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/repair")
async def repair_data(req: RepairRequest):
    try:
        df = pd.DataFrame(req.data)
        engine = SyntheticRepairEngine(req.protected_col, req.cat_cols)
        repaired_df = engine.generate_fair_data(df, req.target)
        return repaired_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))