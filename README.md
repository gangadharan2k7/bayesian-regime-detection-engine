# Bayesian Regime Detection Engine for Equity Direction Forecasting
**Role:** Financial Data Analyst (Kotak Mahindra Asset Management Company Limited)  
**Project:** 1A — Bayesian Deep Learning | Regime Detection | Indian Equity Markets  

## Architecture Overview
- **Data Pipeline:** Point-in-time NSE/RBI/AMFI market & flow ingestion (2007–present)
- **Model Stack:** Frequentist HMM, Bayesian HMM (PyMC), Multivariate Bayesian RS-VAR, Bayesian Deep Learning (MC Dropout, VI, Deep Ensemble), Time-Series Foundation Models (Chronos + TimesFM/Moirai)
- **Online Inference & Calibration:** Bootstrap Particle Filter, BOCPD, Conformal Prediction (Split, APS, Adaptive Conformal Inference)
- **Ensembling & Allocation:** Bayesian Model Averaging (BMA), Time-Aware Stacking, Regime-Conditioned Monte Carlo & Backtesting Overlay
