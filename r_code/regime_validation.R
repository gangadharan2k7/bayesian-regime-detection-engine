ens <- read.csv("artifacts/ensemble_regimes.csv")
bt <- read.csv("artifacts/backtest_timeseries.csv")
cat("Total PIT Sessions:", nrow(ens), "\n")
print(summary(bt[, c("strat_cum", "bench_cum", "equity_weight")]))
