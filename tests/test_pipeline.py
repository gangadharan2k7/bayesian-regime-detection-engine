import json, pathlib, pandas as pd

def test_artifacts_exist():
    for f in [
        'artifacts/ensemble_regimes.csv',
        'artifacts/backtest_report.json',
        'artifacts/conformal_shap_report.json',
        'reports/Project_1A_Bayesian_Regime_Detection_Report.pdf',
    ]:
        assert pathlib.Path(f).exists(), f'Missing {f}'

def test_probabilities_sum_to_one():
    df = pd.read_csv('artifacts/ensemble_regimes.csv')
    probs = df[[f'ensemble_p_{i}' for i in range(5)]].sum(axis=1)
    assert ((probs - 1.0).abs() < 1e-3).all()

def test_latest_contract():
    bt = json.load(open('artifacts/backtest_report.json'))
    c = bt['latest_regime_contract']
    assert c['dominant_regime'] in {'Risk-Off', 'Post-Shock', 'Transitional', 'Late-Cycle', 'Risk-On'}
    assert 0.0 <= c['recommended_equity_weight'] <= 1.0
