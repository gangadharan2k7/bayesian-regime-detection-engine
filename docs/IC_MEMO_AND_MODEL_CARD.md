# INVESTMENT COMMITTEE & SEBI GOVERNANCE REPORT
## Project 1A: Bayesian Regime Detection Engine for Indian Equity Direction Forecasting
**Author:** Gangadharan R | **Engine:** ENSEMBLE_v3.2 | **Date:** 2026-11-23

### 1. Executive Regime Call & Risk-O-Meter Summary
- **Dominant Regime:** **Risk-On** (Meta-Confidence: **56.29/100**)
- **Posterior Probabilities:** {"Risk-Off": 0.0134, "Post-Shock": 0.0276, "Transitional": 0.017, "Late-Cycle": 0.3605, "Risk-On": 0.5815}
- **90% Conformal Set:** `Risk-On` (Empirical OOS Coverage: **100.00%**)
- **Recommended Equity Weight:** **94.6%** | **SEBI Risk-O-Meter:** **Moderate**
- **1-Year Regime-Conditioned MC (10,000 Paths):** Median **17.1%**, 5% VaR **-7.8%**, 95% CVaR **-13.6%**

### 2. Model Stack & Diagnostics
- **Bayesian 5-State HMM (PyMC NUTS):** Max R-hat = `1.0020`, Min ESS = `1803.9`, Divergences = `0`
- **Multivariate Bayesian RS-VAR(1) (NumPyro NUTS):** Max R-hat = `1.0333`, Min ESS = `163.0`
- **6-Model Stacking Weights:** `{"Bayesian_HMM": 0.0, "Bayesian_RSVAR": 0.6567, "Particle_Filter": 0.093, "MC_Dropout_BNN": 0.2503, "Chronos_Foundation": 0.0, "LagLlama_Foundation": 0.0}`
- **OOS Drawdown Control:** Overlay Max DD `-28.01%` vs Benchmark `-38.44%` (Deflated Sharpe Prob: `0.5217`)
