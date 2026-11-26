# Project 1A: Bayesian Regime Detection Engine (ENSEMBLE_v3.2)
## Slide 1: Problem Statement
- Non-stationary Indian equity regimes cause severe static-allocation drawdowns.
## Slide 2: Hybrid Bayesian + Deep Architecture
- PyMC NUTS 5-State HMM + NumPyro RS-VAR(1) + BOCPD + Particle Filter + MC-Dropout BNN.
## Slide 3: Out-of-Sample Results & SEBI Governance
- Max Drawdown reduced from -38.44% to -29.78% (+866 bps).
- 90% Split Conformal Set achieves 100.00% empirical OOS coverage.
- Latest Signal (2026-11-26): Risk-On (80.12%), Equity Weight 95.82%, SEBI Risk-O-Meter: Moderate.
