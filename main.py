import numpy as np
import pandas as pd
import json
import time
from engine import (
    simulate_polymarket_data, build_features, FEATURE_COLS,
    EnsembleSignalEngine, walk_forward_backtest,
    generate_live_signals, get_feature_importance_df
)

SEED = 42
np.random.seed(SEED)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run():
    print_section("POLYSIGNAL — Polymarket ML Prediction Engine")
    print("Generating market simulation data...")

    t0 = time.time()
    raw_df = simulate_polymarket_data(n_markets=80, n_snapshots=200)
    df_feat = build_features(raw_df)

    print(f"Markets: {raw_df['market_id'].nunique()}")
    print(f"Total snapshots: {len(raw_df):,}")
    print(f"Feature-engineered rows: {len(df_feat):,}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Data generation time: {time.time() - t0:.2f}s")

    print_section("Walk-Forward Backtest (5-Fold Time-Series CV)")
    t1 = time.time()
    results = walk_forward_backtest(df_feat, n_splits=5)

    fold_df = results["fold_metrics"]
    print(fold_df[["fold", "accuracy", "roc_auc", "log_loss", "total_pnl", "n_trades", "win_rate"]].to_string(index=False))

    print(f"\n--- Aggregate Performance ---")
    print(f"Mean Accuracy:     {results['mean_accuracy']:.4f}")
    print(f"Mean ROC-AUC:      {results['mean_auc']:.4f}")
    print(f"Mean Log-Loss:     {results['mean_log_loss']:.4f}")
    print(f"Total PnL:         {results['total_pnl']:.4f}")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.4f}")
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Mean Win Rate:     {results['mean_win_rate']:.4f}")
    print(f"Backtest time:     {time.time() - t1:.2f}s")

    print_section("Training Final Model on Full Dataset")
    t2 = time.time()
    X_full = df_feat[FEATURE_COLS].values
    y_full = df_feat["outcome"].values
    final_engine = EnsembleSignalEngine()
    final_engine.fit(X_full, y_full)
    print(f"Final model trained in {time.time() - t2:.2f}s")

    print_section("Live Signal Generation")
    signals_df = generate_live_signals(final_engine, df_feat)
    buy_signals = signals_df[signals_df["signal"] == "BUY"]
    sell_signals = signals_df[signals_df["signal"] == "SELL"]
    hold_signals = signals_df[signals_df["signal"] == "HOLD"]

    print(f"Total markets evaluated: {len(signals_df)}")
    print(f"BUY signals:  {len(buy_signals)}")
    print(f"SELL signals: {len(sell_signals)}")
    print(f"HOLD:         {len(hold_signals)}")

    print("\nTop 10 BUY signals (highest edge):")
    top_buys = buy_signals.nlargest(10, "edge")[
        ["market_id", "category", "current_price", "predicted_prob", "edge", "kelly_fraction"]
    ]
    print(top_buys.to_string(index=False))

    print("\nTop 5 SELL signals (most overpriced):")
    top_sells = sell_signals.nsmallest(5, "edge")[
        ["market_id", "category", "current_price", "predicted_prob", "edge", "kelly_fraction"]
    ]
    print(top_sells.to_string(index=False))

    print_section("Feature Importance (Ensemble-Weighted)")
    imp_df = get_feature_importance_df(final_engine)
    print(imp_df.head(15).to_string(index=False))

    print_section("Saving Results")
    output = {
        "backtest": {
            "mean_accuracy": round(results["mean_accuracy"], 4),
            "mean_roc_auc": round(results["mean_auc"], 4),
            "mean_log_loss": round(results["mean_log_loss"], 4),
            "total_pnl": round(results["total_pnl"], 4),
            "sharpe_ratio": round(results["sharpe_ratio"], 4),
            "max_drawdown": round(results["max_drawdown"], 4),
            "total_trades": int(results["total_trades"]),
            "mean_win_rate": round(results["mean_win_rate"], 4),
        },
        "signals_summary": {
            "buy": int(len(buy_signals)),
            "sell": int(len(sell_signals)),
            "hold": int(len(hold_signals)),
        },
        "top_buy_signals": top_buys.to_dict(orient="records"),
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    signals_df.to_csv("signals.csv", index=False)
    imp_df.to_csv("feature_importance.csv", index=False)

    print("results.json saved")
    print("signals.csv saved")
    print("feature_importance.csv saved")
    print(f"\nTotal runtime: {time.time() - t0:.2f}s")
    print_section("DONE")


if __name__ == "__main__":
    run()
