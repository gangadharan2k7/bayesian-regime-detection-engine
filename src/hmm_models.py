import json
import warnings
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from hmmlearn import hmm
from scipy import stats

warnings.filterwarnings("ignore")

REGIME_ORDER = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]

def fit_frequentist_hmm(returns: pd.Series, n_states: int = 5, n_iter: int = 300, seed: int = 42):
    X = returns.values.reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        tol=1e-5,
    )
    model.fit(X)
    raw_states = model.predict(X)
    raw_probs = model.predict_proba(X)

    # Order states from Risk-Off (lowest return / highest vol) to Risk-On (highest return / lowest vol)
    summary = []
    for i in range(n_states):
        mu = float(model.means_[i, 0])
        sig = float(np.sqrt(model.covars_[i, 0, 0]))
        score = mu / (sig + 1e-6)
        summary.append((i, mu, sig, score))

    summary.sort(key=lambda x: x[3])  # ascending: Risk-Off (0) -> Risk-On (4)
    perm = [x[0] for x in summary]

    ordered_probs = raw_probs[:, perm]
    inv_map = {old_idx: new_idx for new_idx, old_idx in enumerate(perm)}
    ordered_states = np.array([inv_map[s] for s in raw_states])
    ordered_transmat = model.transmat_[np.ix_(perm, perm)]
    ordered_means = model.means_[perm, 0]
    ordered_stds = np.sqrt(model.covars_[perm, 0, 0])

    return {
        "model": model,
        "states": ordered_states,
        "probs": ordered_probs,
        "transmat": ordered_transmat,
        "means": ordered_means,
        "stds": ordered_stds,
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "loglik": float(model.score(X)),
    }

def compare_hmm_states_bic(returns: pd.Series) -> dict:
    res = {}
    for k in [3, 5, 7]:
        X = returns.values.reshape(-1, 1)
        m = hmm.GaussianHMM(n_components=k, covariance_type="diag", n_iter=250, random_state=42)
        m.fit(X)
        res[f"K={k}"] = {"loglik": float(m.score(X)), "aic": float(m.aic(X)), "bic": float(m.bic(X))}
    return res

def test_geometric_duration_null(states: np.ndarray, K: int = 5) -> dict:
    """Test observed regime run-lengths against geometric null (Section A15.3)."""
    durations = []
    cur = states[0]
    length = 1
    for s in states[1:]:
        if s == cur:
            length += 1
        else:
            durations.append(length)
            cur = s
            length = 1
    durations.append(length)
    durations = np.array(durations)
    p_hat = 1.0 / np.mean(durations)
    ks_stat, p_val = stats.kstest(durations, "geom", args=(p_hat,))
    return {
        "mean_duration_days": float(np.mean(durations)),
        "median_duration_days": float(np.median(durations)),
        "geometric_p_hat": float(p_hat),
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "reject_geometric_null": bool(p_val < 0.05),
    }

def fit_bayesian_hmm_pymc(returns: np.ndarray, init_states: np.ndarray, K: int = 5):
    """Bayesian Gaussian-emission HMM with sticky Dirichlet transition priors & MCMC diagnostics (Section A3.4 & A15.2)."""
    # Use trailing 1,250 trading days (~5 years) for fast, exact NUTS posterior sampling
    r_sub = returns[-1250:]
    z_sub = init_states[-1250:]

    # Empirical transition counts for Dirichlet posterior + sticky diagonal prior
    trans_counts = np.zeros((K, K))
    for t in range(len(z_sub) - 1):
        trans_counts[z_sub[t], z_sub[t + 1]] += 1.0

    alpha_prior = np.ones((K, K)) * 1.0 + np.eye(K) * 8.0

    with pm.Model() as model:
        P = pm.Dirichlet("P", a=alpha_prior + trans_counts, shape=(K, K))
        mu = pm.Normal(
            "mu",
            mu=np.array([-0.0025, -0.0005, 0.0001, 0.0006, 0.0012]),
            sigma=0.002,
            shape=K,
        )
        sigma = pm.HalfNormal("sigma", sigma=0.025, shape=K)

        # Regime-conditioned emission likelihood
        obs = pm.Normal("obs", mu=mu[z_sub], sigma=sigma[z_sub], observed=r_sub)

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.95,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    summary = az.summary(trace, var_names=["mu", "sigma", "P"])
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    divergences = int(trace.sample_stats["diverging"].sum().values)

    # Posterior predictive regime probabilities across full history
    mu_post = trace.posterior["mu"].mean(dim=("chain", "draw")).values
    sig_post = trace.posterior["sigma"].mean(dim=("chain", "draw")).values
    P_post = trace.posterior["P"].mean(dim=("chain", "draw")).values

    T = len(returns)
    filt_probs = np.zeros((T, K))
    pi_cur = np.ones(K) / K
    for t in range(T):
        pred_pi = pi_cur @ P_post
        lik = stats.norm.pdf(returns[t], loc=mu_post, scale=sig_post) + 1e-12
        post_pi = pred_pi * lik
        post_pi /= post_pi.sum()
        filt_probs[t] = post_pi
        pi_cur = post_pi

    diagnostics = {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": divergences,
        "P_post_mean": P_post.tolist(),
        "mu_post_mean": mu_post.tolist(),
        "sigma_post_mean": sig_post.tolist(),
    }
    return trace, filt_probs, diagnostics

if __name__ == "__main__":
    df = pd.read_csv("data/indian_market_pit.csv", parse_dates=["Date"], index_col="Date")
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    df = df.loc[feat.index]
    returns = df["NiftyReturn"]

    print("1. Fitting Frequentist 5-State Gaussian HMM & BIC model selection...")
    freq_res = fit_frequentist_hmm(returns, n_states=5)
    bic_comp = compare_hmm_states_bic(returns)
    dur_test = test_geometric_duration_null(freq_res["states"], K=5)

    print("2. Sampling Bayesian HMM via PyMC NUTS (4 chains, 1000 draws)...")
    trace, bayes_probs, mcmc_diag = fit_bayesian_hmm_pymc(returns.values, freq_res["states"], K=5)
    trace.to_netcdf("artifacts/bayesian_hmm_trace.nc")

    out_probs = pd.DataFrame(index=feat.index)
    out_probs["hmm_freq_state"] = freq_res["states"]
    for k, name in enumerate(REGIME_ORDER):
        out_probs[f"hmm_freq_p_{k}"] = freq_res["probs"][:, k]
        out_probs[f"hmm_bayes_p_{k}"] = bayes_probs[:, k]
    out_probs.to_csv("artifacts/hmm_regime_probs.csv")

    report = {
        "bic_comparison": bic_comp,
        "duration_hypothesis_test": dur_test,
        "bayesian_hmm_mcmc_diagnostics": mcmc_diag,
    }
    with open("artifacts/hmm_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"MCMC Diagnostics -> max R-hat: {mcmc_diag['max_rhat']:.4f}, min ESS: {mcmc_diag['min_ess_bulk']:.1f}, divergences: {mcmc_diag['divergences']}")
    print(f"Duration Test -> Mean: {dur_test['mean_duration_days']:.1f}d, Reject Geometric Null: {dur_test['reject_geometric_null']}")
