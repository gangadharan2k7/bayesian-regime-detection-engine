# INVESTMENT COMMITTEE & SEBI GOVERNANCE REPORT
## Project 1A: Bayesian Regime Detection Engine for Indian Equity Direction Forecasting
**Author:** Gangadharan R | **Engine:** ENSEMBLE_v3.2 | **Date:** 2026-11-26

### 1. Executive Regime Call & Risk-O-Meter Summary
- **Dominant Regime:** **Risk-On** (Meta-Confidence: **73.72/100**)
- **Posterior Probabilities:** {"Risk-Off": 0.0053, "Post-Shock": 0.0074, "Transitional": 0.0239, "Late-Cycle": 0.1623, "Risk-On": 0.8012}
- **90% Conformal Set:** `Risk-On` (Empirical OOS Coverage: **100.00%**)
- **Recommended Equity Weight:** **95.8%** | **SEBI Risk-O-Meter:** **Moderate**
- **1-Year Regime-Conditioned MC (10,000 Paths):** Median **17.1%**, 5% VaR **-7.9%**, 95% CVaR **-13.6%**

### 2. Model Stack & Diagnostics
- **Bayesian 5-State HMM (PyMC NUTS):** Max R-hat = `1.0033`, Min ESS = `1952.5`, Divergences = `0`
- **Multivariate Bayesian RS-VAR(1) (NumPyro NUTS):** Max R-hat = `1.0161`, Min ESS = `340.7`
- **6-Model Stacking Weights:** `{"Bayesian_HMM": 0.0, "Bayesian_RSVAR": 0.4727, "Particle_Filter": 0.0656, "MC_Dropout_BNN": 0.4617, "Chronos_Foundation": 0.0, "LagLlama_Foundation": 0.0}`
- **OOS Drawdown Control:** Overlay Max DD `-29.78%` vs Benchmark `-38.44%` (Deflated Sharpe Prob: `0.4901`)
