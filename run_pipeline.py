"""
End-to-End Orchestrator for Project 1A: Bayesian Regime Detection Engine (ENSEMBLE_v3.2)
Runs data ingestion, feature engineering, Bayesian/Deep regime models, conformal SHAP, and backtesting.
"""
import subprocess
import sys
from pathlib import Path

PIPELINE_STEPS = [
    "src/data_pipeline.py",
    "src/features.py",
    "src/hmm_models.py",
    "src/rsvar_model.py",
    "src/filtering_changepoints.py",
    "src/deep_foundation_models.py",
    "src/conformal_shap.py",
    "src/backtest_portfolio.py",
    "src/make_pdf_report.py",
]

def main() -> None:
    root = Path(__file__).resolve().parent
    for step in PIPELINE_STEPS:
        script_path = root / step
        if not script_path.exists():
            print(f"[SKIP] {step} not found.")
            continue
        print(f"\n=== Running {step} ===")
        result = subprocess.run([sys.executable, str(script_path)], cwd=root)
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed at {step} with exit code {result.returncode}")
    print("\n[SUCCESS] Full ENSEMBLE_v3.2 pipeline and artifacts generated.")

if __name__ == "__main__":
    main()
