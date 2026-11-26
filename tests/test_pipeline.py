import pathlib, pandas as pd
def test_all():
 assert pathlib.Path('reports/Project_1A_Bayesian_Regime_Detection_Report.pdf').exists()
 df=pd.read_csv('artifacts/ensemble_regimes.csv')
 assert ((df[[f'ensemble_p_{i}' for i in range(5)]].sum(axis=1)-1).abs()<1e-3).all()
