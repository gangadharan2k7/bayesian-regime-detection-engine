import json
from datetime import datetime
from fastapi import FastAPI

app = FastAPI(title="Bayesian Regime Detection Engine API", version="3.2.0")

@app.get("/regime/health")
def health():
    with open("artifacts/data_lineage.json", "r", encoding="utf-8") as f:
        lineage = json.load(f)
    with open("artifacts/conformal_shap_report.json", "r", encoding="utf-8") as f:
        cp = json.load(f)
    return {
        "status": "HEALTHY",
        "engine_version": "ENSEMBLE_v3.2",
        "data_snapshot_sha256": lineage["sha256"][:16],
        "last_trading_date": lineage["end_date"],
        "conformal_oos_coverage": cp["conformal_calibration"]["oos_empirical_coverage"],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/regime/score")
def latest_score():
    with open("artifacts/backtest_report.json", "r", encoding="utf-8") as f:
        bt = json.load(f)
    with open("artifacts/conformal_shap_report.json", "r", encoding="utf-8") as f:
        cp = json.load(f)
    return {
        "regime_contract": bt["latest_regime_contract"],
        "top_shap_drivers": list(cp["top_global_shap_drivers"].items())[:5],
        "monte_carlo_1y_outlook": bt["regime_conditioned_mc_1y"],
    }
