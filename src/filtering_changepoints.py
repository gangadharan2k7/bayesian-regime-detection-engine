import json
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

REGIME_ORDER = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]

def run_bocpd(returns: np.ndarray, hazard: float = 1.0 / 125.0, max_run: int = 300):
    """Adams & MacKay (2007) Bayesian Online Changepoint Detection with Normal-Inverse-Gamma conjugate updates."""
    T = len(returns)
    R = np.zeros((T + 1, max_run + 1))
    R[0, 0] = 1.0

    mu0, kappa0, alpha0, beta0 = 0.0, 0.1, 2.0, 0.0002
    muT = np.array([mu0])
    kappaT = np.array([kappa0])
    alphaT = np.array([alpha0])
    betaT = np.array([beta0])

    cp_prob = np.zeros(T)
    exp_run_len = np.zeros(T)

    for t in range(T):
        x = returns[t]
        df = 2.0 * alphaT
        scale = np.sqrt(betaT * (kappaT + 1.0) / (alphaT * kappaT))
        pred_probs = stats.t.pdf(x, df=df, loc=muT, scale=scale) + 1e-15

        growth_probs = R[t, : len(muT)] * pred_probs * (1.0 - hazard)
        cp_mass = np.sum(R[t, : len(muT)] * pred_probs * hazard)

        new_R = np.zeros(max_run + 1)
        new_R[0] = cp_mass
        keep = min(len(growth_probs), max_run)
        new_R[1 : keep + 1] = growth_probs[:keep]
        new_R /= new_R.sum() + 1e-15
        R[t + 1] = new_R

        # Short-run mass (run length <= 5) captures acute regime breaks
        cp_prob[t] = float(np.sum(new_R[:6]))
        exp_run_len[t] = float(np.sum(np.arange(max_run + 1) * new_R))

        # Conjugate NIG hyperparameter update
        mu_next = (kappaT * muT + x) / (kappaT + 1.0)
        kappa_next = kappaT + 1.0
        alpha_next = alphaT + 0.5
        beta_next = betaT + 0.5 * kappaT * ((x - muT) ** 2) / (kappaT + 1.0)

        muT = np.concatenate([[mu0], mu_next[:max_run]])
        kappaT = np.concatenate([[kappa0], kappa_next[:max_run]])
        alphaT = np.concatenate([[alpha0], alpha_next[:max_run]])
        betaT = np.concatenate([[beta0], beta_next[:max_run]])

    return cp_prob, exp_run_len

def run_offline_pelt(feat: pd.DataFrame, pen: float = 18.0) -> list:
    """Offline PELT changepoint baseline via ruptures (Section A5.2)."""
    sig = feat[["ret_21d", "vol_21d", "vix_z"]].values
    sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
    algo = rpt.Pelt(model="rbf", min_size=21, jump=5).fit(sig)
    bkps = algo.predict(pen=pen)
    dates = [str(feat.index[min(b - 1, len(feat) - 1)].date()) for b in bkps[:-1]]
    return dates

def run_particle_filter(feat: pd.DataFrame, n_particles: int = 1000, seed: int = 42):
    """Bootstrap Particle Filter with systematic resampling & N_eff diagnostics (Section A6.3)."""
    rng = np.random.default_rng(seed)
    K = 5
    T = len(feat)

    # Regime emission prototypes: [ret_21d, vix_z, pct_above_50dma, fii_flow_z_60d, usdinr_z_60d]
    obs_cols = ["ret_21d", "vix_z", "pct_above_50dma", "fii_flow_z_60d", "usdinr_z_60d"]
    Y = feat[obs_cols].values.copy()
    Y[:, 2] = (Y[:, 2] - 50.0) / 25.0  # normalise breadth

    prototypes = np.array([
        [-0.08,  2.2, -1.2, -1.5,  1.4],  # 0: Risk-Off
        [ 0.04,  1.1,  0.1,  0.2,  0.3],  # 1: Post-Shock
        [ 0.00,  0.1,  0.0, -0.2,  0.1],  # 2: Transitional
        [ 0.02, -0.3,  0.2, -0.3,  0.2],  # 3: Late-Cycle
        [ 0.04, -0.8,  0.9,  0.8, -0.6],  # 4: Risk-On
    ])
    scales = np.array([0.05, 1.0, 0.7, 1.0, 0.9])

    P = np.array([
        [0.86, 0.10, 0.03, 0.01, 0.00],
        [0.05, 0.84, 0.06, 0.01, 0.04],
        [0.04, 0.03, 0.85, 0.04, 0.04],
        [0.05, 0.01, 0.06, 0.84, 0.04],
        [0.01, 0.01, 0.04, 0.05, 0.89],
    ])

    particles = rng.integers(0, K, size=n_particles)
    weights = np.ones(n_particles) / n_particles

    pf_probs = np.zeros((T, K))
    ess_series = np.zeros(T)

    for t in range(T):
        # Propagate via Markov transition
        u = rng.random(n_particles)
        cdf = np.cumsum(P[particles], axis=1)
        particles = (u[:, None] > cdf).sum(axis=1)

        # Observation log-likelihood
        diff = (Y[t][None, :] - prototypes[particles]) / scales[None, :]
        log_w = -0.5 * np.sum(diff ** 2, axis=1)
        log_w -= log_w.max()
        weights = np.exp(log_w) + 1e-12
        weights /= weights.sum()

        for k in range(K):
            pf_probs[t, k] = np.sum(weights[particles == k])

        n_eff = 1.0 / np.sum(weights ** 2)
        ess_series[t] = n_eff

        # Systematic resampling if N_eff < N/2
        if n_eff < 0.5 * n_particles:
            positions = (rng.random() + np.arange(n_particles)) / n_particles
            cum_w = np.cumsum(weights)
            idx = np.searchsorted(cum_w, positions)
            particles = particles[idx]
            weights = np.ones(n_particles) / n_particles

    return pf_probs, ess_series

if __name__ == "__main__":
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    hmm_df = pd.read_csv("artifacts/hmm_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    rsv_df = pd.read_csv("artifacts/rsvar_regime_probs.csv", parse_dates=["Date"], index_col="Date")

    print("1. Running Bayesian Online Changepoint Detection (BOCPD) & Offline PELT...")
    cp_prob, run_len = run_bocpd(feat["ret_1d"].values)
    pelt_dates = run_offline_pelt(feat)

    print("2. Running Bootstrap Particle Filter (N=1000 particles)...")
    pf_probs, ess_series = run_particle_filter(feat, n_particles=1000)

    # Wasserstein-1 distance between Particle Filter and RS-VAR state distributions
    rsv_mat = rsv_df[[f"rsvar_p_{k}" for k in range(5)]].values
    hmm_mat = hmm_df[[f"hmm_bayes_p_{k}" for k in range(5)]].values
    states_grid = np.arange(5)
    w1_dists = [
        float(stats.wasserstein_distance(states_grid, states_grid, pf_probs[t], rsv_mat[t]))
        for t in range(len(feat))
    ]

    # Calibrated Regime Ensemble Posterior (0.30 Bayesian HMM + 0.35 RS-VAR + 0.35 Particle Filter)
    ens_probs = 0.30 * hmm_mat + 0.35 * rsv_mat + 0.35 * pf_probs
    ens_probs /= ens_probs.sum(axis=1, keepdims=True)

    # Regime Stability Score: 1 - H(p_t) / log(5)
    entropy = -np.sum(ens_probs * np.log(ens_probs + 1e-12), axis=1)
    stability = 1.0 - entropy / np.log(5.0)

    out = pd.DataFrame(index=feat.index)
    out["bocpd_cp_prob"] = cp_prob
    out["bocpd_run_length"] = run_len
    out["pf_ess"] = ess_series
    out["wasserstein_pf_rsvar"] = w1_dists
    out["regime_stability_score"] = stability
    out["ensemble_regime_id"] = np.argmax(ens_probs, axis=1)
    out["ensemble_regime_label"] = [REGIME_ORDER[i] for i in out["ensemble_regime_id"]]

    for k in range(5):
        out[f"pf_p_{k}"] = pf_probs[:, k]
        out[f"ensemble_p_{k}"] = ens_probs[:, k]

    out.to_csv("artifacts/ensemble_regimes.csv")

    summary = {
        "pelt_changepoint_dates": pelt_dates,
        "pelt_count": len(pelt_dates),
        "mean_pf_ess": float(np.mean(ess_series)),
        "mean_wasserstein_pf_rsvar": float(np.mean(w1_dists)),
        "regime_distribution_pct": (out["ensemble_regime_label"].value_counts(normalize=True) * 100).round(2).to_dict(),
    }
    with open("artifacts/filtering_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Detected {len(pelt_dates)} PELT structural breaks | Mean PF ESS: {summary['mean_pf_ess']:.1f} | Mean W1: {summary['mean_wasserstein_pf_rsvar']:.4f}")
    print("Ensemble Regime Distribution (%):", summary["regime_distribution_pct"])
