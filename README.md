# Project 1A: Bayesian Regime Detection Engine (ENSEMBLE_v3.2)

**Author:** Gangadharan R | **Program:** ZeTheta Algorithms Private Limited

## Out-of-Sample Results (2007-2026, 4,682 PIT Sessions)
- **Latest Signal (2026-11-26):** Risk-On (80.12% posterior) | Meta-Confidence: 73.72/100
- **Conformal Prediction Set (90% Target):** {Risk-On} (100.00% OOS coverage)
- **Recommended Equity Weight:** 95.82% (SEBI Risk-O-Meter: Moderate)
- **Max Drawdown:** -29.78% vs Benchmark -38.44% (+866 bps improvement)
- **Annualized Volatility:** 13.39% vs Benchmark 16.82%

## Quickstart
`ash
pip install -r requirements.txt
python main.py
pytest
streamlit run app/dashboard.py
`
