# Project 1A: Bayesian Regime Detection Engine (ENSEMBLE_v3.2)

**Author:** Gangadharan R  
**Program:** ZeTheta Algorithms Private Limited - Project Work Experience Program  
**Executive PDF Report:** [Project_1A_Bayesian_Regime_Detection_Report.pdf](./Project_1A_Bayesian_Regime_Detection_Report.pdf)  
**Model Card & IC Memo:** [docs/IC_MEMO_AND_MODEL_CARD.md](./docs/IC_MEMO_AND_MODEL_CARD.md)

## Overview
End-to-end probabilistic regime detection and dynamic asset allocation engine built on 4,682 Point-in-Time (PIT) trading sessions (2007-09-24 to 2026-11-26) with 43 leak-free engineered features.

## Key Out-of-Sample Results
- **Latest Evaluation Date:** 2026-11-26
- **Dominant Regime:** Risk-On (80.12% posterior probability)
- **Conformal Prediction Set (90% Target):** {Risk-On} (100.00% empirical OOS coverage)
- **Meta-Confidence Score:** 73.72 / 100
- **Recommended Equity Weight:** 95.82% (SEBI Risk-O-Meter: Moderate)
- **OOS Max Drawdown:** -29.78% vs Benchmark -38.44% (+866 bps improvement)
- **OOS Annualized Volatility:** 13.39% vs Benchmark 16.82%

## Quickstart
`ash
pip install -r requirements.txt
python run_pipeline.py
python src/api_server.py
streamlit run app/dashboard.py
`
