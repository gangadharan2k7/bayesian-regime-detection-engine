import json
import warnings
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

REGIME_ORDER = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]

def fit_msm_baseline(returns: np.ndarray, k_regimes: int = 3) -> dict:
    mod = MarkovRegression(returns[-1500:], k_regimes=k_regimes, trend="c", switching_variance=True)
    res = mod.fit(disp=False)
    return {
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llf": float(res.llf),
        "params": {k: float(v) for k, v in zip(res.model.param_names, res.params)},
    }

def fit_bayesian_rsvar(Y_df: pd.DataFrame, states: np.ndarray, K: int = 5):
    """Hierarchical Bayesian RS-VAR(1) via NumPyro XLA NUTS (fast ~25s convergence)."""
    cols = Y_df.columns.tolist()
    d = len(cols)

    Y_mean = Y_df.mean().values
    Y_std = Y_df.std().values + 1e-8
    Y_norm = (Y_df.values - Y_mean) / Y_std

    Y_sub = Y_norm[-750:]
    z_sub = states[-750:]

    Y_t = Y_sub[1:]
    Y_lag = Y_sub[:-1]
    z_t = z_sub[1:]

    with pm.Model() as model:
        c_global = pm.Normal("c_global", mu=0.0, sigma=0.25, shape=d)
        c_offset = pm.Normal("c_offset", mu=0.0, sigma=0.25, shape=(K, d))
        c = pm.Deterministic("c", c_global + c_offset)

        A_global = pm.Normal("A_global", mu=0.0, sigma=0.2, shape=(d, d))
        A_offset = pm.Normal("A_offset", mu=0.0, sigma=0.15, shape=(K, d, d))
        A = pm.Deterministic("A", A_global + A_offset)

        sigma = pm.HalfNormal("sigma", sigma=0.8, shape=(K, d))

        mu = c[z_t] + pm.math.sum(A[z_t] * Y_lag[:, None, :], axis=-1)
        pm.Normal("obs", mu=mu, sigma=sigma[z_t], observed=Y_t)

        trace = pm.sample(
            draws=300,
            tune=300,
            chains=4,
            target_accept=0.90,
            random_seed=42,
            nuts_sampler="numpyro",
            progressbar=True,
            idata_kwargs={"log_likelihood": True},
        )

    summary = az.summary(trace, var_names=["c_global", "A_global", "sigma"])
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    divergences = int(trace.sample_stats["diverging"].sum().values)

    c_post = trace.posterior["c"].mean(dim=("chain", "draw")).values
    A_post = trace.posterior["A"].mean(dim=("chain", "draw")).values
    sig_post = trace.posterior["sigma"].mean(dim=("chain", "draw")).values

    T = len(Y_norm)
    rsvar_probs = np.zeros((T, K))
    rsvar_probs[0] = np.ones(K) / K

    P_mat = np.ones((K, K)) * 0.02 + np.eye(K) * 0.90
    P_mat /= P_mat.sum(axis=1, keepdims=True)

    for t in range(1, T):
        pred_pi = rsvar_probs[t - 1] @ P_mat
        log_lik = np.zeros(K)
        for k in range(K):
            mean_k = c_post[k] + A_post[k] @ Y_norm[t - 1]
            log_lik[k] = np.sum(stats.norm.logpdf(Y_norm[t], loc=mean_k, scale=sig_post[k]))
        log_lik -= log_lik.max()
        lik = np.exp(log_lik) + 1e-12
        post_pi = pred_pi * lik
        rsvar_probs[t] = post_pi / post_pi.sum()

    fii_idx = cols.index("fii_flow_z_60d")
    ret_idx = cols.index("ret_1d")
    irf = {}
    for k, rname in enumerate(REGIME_ORDER):
        shock = np.zeros(d)
        shock[fii_idx] = -1.0
        path = []
        cur = shock.copy()
        for h in range(10):
            path.append(float(cur[ret_idx] * Y_std[ret_idx]))
            cur = A_post[k] @ cur
        irf[rname] = path

    cov_report = {}
    for k, rname in enumerate(REGIME_ORDER):
        mask = z_sub == k
        if mask.sum() > 10:
            corr_k = np.corrcoef(Y_sub[mask], rowvar=False)
            cov_report[rname] = np.round(np.nan_to_num(corr_k), 3).tolist()
        else:
            cov_report[rname] = np.eye(d).tolist()

    diagnostics = {
        "variables": cols,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": divergences,
        "fii_outflow_irf_10d": irf,
        "regime_conditional_correlations": cov_report,
    }
    return trace, rsvar_probs, diagnostics

if __name__ == "__main__":
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    hmm_df = pd.read_csv("artifacts/hmm_regime_probs.csv", parse_dates=["Date"], index_col="Date")

    print("1. Fitting statsmodels Markov-Switching regression baseline...")
    msm_res = fit_msm_baseline(feat["ret_1d"].values, k_regimes=3)

    print("2. Sampling Hierarchical Multivariate Bayesian RS-VAR(1) via NumPyro XLA NUTS...")
    rsv_cols = ["ret_1d", "vix_z", "pct_above_50dma", "fii_flow_z_60d", "usdinr_z_60d"]
    trace, rsvar_probs, diag = fit_bayesian_rsvar(feat[rsv_cols], hmm_df["hmm_freq_state"].values, K=5)
    trace.to_netcdf("artifacts/bayesian_rsvar_trace.nc")

    out = pd.DataFrame(index=feat.index)
    for k in range(5):
        out[f"rsvar_p_{k}"] = rsvar_probs[:, k]
    out.to_csv("artifacts/rsvar_regime_probs.csv")

    diag["msm_baseline"] = msm_res
    with open("artifacts/rsvar_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"RS-VAR MCMC -> max R-hat: {diag['max_rhat']:.4f}, min ESS: {diag['min_ess_bulk']:.1f}, divergences: {diag['divergences']}")
