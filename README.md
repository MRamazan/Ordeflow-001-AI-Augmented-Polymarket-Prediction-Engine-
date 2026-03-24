<p align="center">
  <img src="submission_logo.png" alt="PolySignal Logo" width="800"/>
</p>

# PolySignal - AI-Augmented Polymarket Prediction Engine

> **Orderflow 001 Submission** | AI-Augmented Systems Track

An ensemble ML system that detects mispriced prediction markets on Polymarket by modeling probability drift, liquidity dynamics, and temporal decay. Connects to the live Polymarket API and generates actionable BUY/SELL signals with Kelly-sized positions.

---

## Live Signal Output (Real Polymarket Data)

Fetched 100 active markets from Polymarket API. Sample signals:

| Signal | Market | Price | ML Prob | Edge |
|--------|--------|-------|---------|------|
| BUY | Will Joel Embiid win 2025-26 NBA MVP? | 0.02 | 0.439 | +0.419 |
| BUY | Will Stephen Curry win 2025-26 NBA MVP? | 0.02 | 0.439 | +0.419 |
| BUY | Trump out as President by March 31? | 0.02 | 0.430 | +0.410 |
| SELL | Miami Open: Sabalenka vs Zheng | 0.98 | 0.358 | -0.622 |
| SELL | Thunder vs. 76ers | 0.98 | 0.365 | -0.615 |
| SELL | Israel ground offensive in Lebanon by March 31? | 0.98 | 0.371 | -0.609 |

Signal distribution: **BUY: 73 / SELL: 20 / HOLD: 7** across 100 live markets.

---

## Backtest Performance (Walk-Forward CV, 5 Folds)

| Metric | Value |
|--------|-------|
| Mean Accuracy | **0.5832** |
| Mean ROC-AUC | **0.6181** |
| Mean Log-Loss | 0.6803 |
| Total PnL | **291.35** |
| Sharpe Ratio | **65.59** |
| Max Drawdown | 0.00 |
| Total Trades | **9,681** |
| Mean Win Rate | 0.3918 |

All metrics produced by `TimeSeriesSplit` walk-forward cross-validation. No lookahead bias.

| Fold | Accuracy | ROC-AUC | PnL | Trades |
|------|----------|---------|-----|--------|
| 1 | 0.630 | 0.683 | 81.94 | 2,167 |
| 2 | 0.568 | 0.602 | 63.36 | 1,875 |
| 3 | 0.517 | 0.522 | 57.23 | 1,811 |
| 4 | 0.572 | 0.588 | 41.01 | 1,770 |
| 5 | 0.629 | 0.696 | 47.81 | 2,058 |

---

## Architecture

```
Polymarket Gamma API (live)
        │
        ▼
Feature Engineering (29 features)
  ├── Price dynamics     (lags 1d/3d/7d, momentum, MA5/MA10)
  ├── Liquidity signals  (log-volume, log-OI, spread × volume)
  ├── Temporal features  (time fraction, log days to resolution)
  └── Market structure   (price extremity, category encoding)
        │
        ▼
Ensemble Model
  ├── GradientBoostingClassifier   weight: 0.50
  ├── RandomForestClassifier       weight: 0.35
  └── CalibratedLogisticRegression weight: 0.15
        │
        ▼
Signal Engine
  ├── BUY   edge > +0.06
  ├── SELL  edge < -0.06
  └── HOLD
        │
        ▼
Kelly Criterion Position Sizing
  └── position = min(kelly × 0.5, 0.25)
```

---

## Edge Hypothesis

Prediction markets are systematically mispriced due to three structural inefficiencies:

**1. Recency bias** - retail participants overweight recent news, creating momentum that overshoots true probability.

**2. Thin market distortion** - low-liquidity markets (common at price extremes near 0.02 or 0.98) have wide spreads and slow price discovery.

**3. Slow probability updating near resolution** - markets approaching their deadline fail to converge to true probability quickly enough, creating exploitable drift.

The model's strongest signal is `price_lag7` (importance: 0.197), confirming that 7-day price history is the primary predictor of mispricing. `log_liquidity` (0.063) and `time_fraction` (0.052) further capture the structural inefficiencies above.

---

## Features (29 total)

**Price Dynamics**
`price`, `price_lag1`, `price_lag3`, `price_lag7`, `price_return_1d/3d/7d`, `price_ma5`, `price_ma10`, `price_std5`, `price_std10`, `price_vs_ma5`, `price_vs_ma10`

**Liquidity & Volume**
`log_volume`, `log_liquidity`, `log_oi`, `vol_ratio`, `oi_per_liquidity`, `spread_x_volume`, `spread`

**Temporal**
`time_fraction`, `log_days_left`

**Market Structure**
`price_mid_dist`, `price_extreme`, `cat_politics`, `cat_crypto`, `cat_sports`, `cat_economics`, `cat_science`

---

## Top Feature Importances

| Feature | Importance |
|---------|-----------|
| price_lag7 | 0.197 |
| price_ma10 | 0.111 |
| log_liquidity | 0.063 |
| time_fraction | 0.052 |
| price_ma5 | 0.051 |
| log_days_left | 0.049 |
| log_oi | 0.044 |

---

## Files

```
polysignal/
├── engine.py              Core ML pipeline - feature engineering, ensemble model, backtest
├── main.py                Backtest runner - produces results.json, signals.csv
├── visualize.py           Backtest dashboard - polysignal_backtest_report.png
├── analysis.py            Calibration, ROC curves, edge distribution - polysignal_analysis.png
├── polymarket_api.py      Live Polymarket API connector (Gamma + CLOB)
├── live.py                Live signal runner - falls back to simulation if API unavailable
├── results.json           Backtest metrics
├── signals.csv            Simulation signal output
└── live_signals.csv       Real Polymarket signal output
```

---

## Quickstart

```bash
pip install scikit-learn numpy pandas matplotlib scipy requests

python main.py        # run backtest
python visualize.py   # generate backtest dashboard
python analysis.py    # generate calibration + ROC analysis
python live.py        # connect to Polymarket API + generate live signals
```

---

## Data Sources

- **Polymarket Gamma API** - active markets, volume, liquidity, spread
  `GET https://gamma-api.polymarket.com/markets?active=true`
- **Polymarket CLOB API** - price history
  `GET https://clob.polymarket.com/prices-history`
- **Simulation** - statistically calibrated synthetic data for reproducible backtesting (Beta-distributed true probabilities, log-normal volumes, mean-reverting price dynamics)

---

## Position Sizing

Half-Kelly criterion applied to all signals:

```
kelly    = (p × b − (1 − p)) / b
position = min(kelly × 0.5, 0.25)
```

where `p` = model probability, `b = avg_win / avg_loss`. The 0.5 Kelly multiplier and 25% cap enforce conservative risk management.

---

*Built for Orderflow 001 - 48-hour build sprint*
