import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import json
import warnings
warnings.filterwarnings("ignore")

from engine import (
    simulate_polymarket_data, build_features, FEATURE_COLS,
    EnsembleSignalEngine, walk_forward_backtest,
    generate_live_signals, get_feature_importance_df
)

SEED = 42
np.random.seed(SEED)

raw_df = simulate_polymarket_data(n_markets=80, n_snapshots=200)
df_feat = build_features(raw_df)
results = walk_forward_backtest(df_feat, n_splits=5)
X_full = df_feat[FEATURE_COLS].values
y_full = df_feat["outcome"].values
final_engine = EnsembleSignalEngine()
final_engine.fit(X_full, y_full)
signals_df = generate_live_signals(final_engine, df_feat)
imp_df = get_feature_importance_df(final_engine)
fold_df = results["fold_metrics"]

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#e3b341"
TEXT = "#e6edf3"
SUBTEXT = "#8b949e"
BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "axes.edgecolor": BORDER, "figure.facecolor": BG,
})


def card_ax(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)


fig.text(0.5, 0.97, "PolySignal — ML Prediction Engine", ha="center",
         fontsize=20, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945, "Polymarket AI-Augmented Trading System  |  Walk-Forward Backtest Results",
         ha="center", fontsize=11, color=SUBTEXT)


ax_metric = fig.add_subplot(gs[0, :])
ax_metric.set_facecolor(CARD)
ax_metric.axis("off")
for spine in ax_metric.spines.values():
    spine.set_edgecolor(BORDER)

metrics = [
    ("Mean Accuracy", f"{results['mean_accuracy']:.3f}", ACCENT),
    ("Mean ROC-AUC", f"{results['mean_auc']:.3f}", GREEN),
    ("Sharpe Ratio", f"{results['sharpe_ratio']:.1f}", YELLOW),
    ("Total PnL", f"{results['total_pnl']:.1f}", GREEN),
    ("Total Trades", f"{results['total_trades']:,}", ACCENT),
    ("Mean Win Rate", f"{results['mean_win_rate']:.3f}", YELLOW),
]
for i, (label, val, color) in enumerate(metrics):
    x = 0.08 + i * 0.155
    ax_metric.text(x, 0.72, val, transform=ax_metric.transAxes,
                   fontsize=22, fontweight="bold", color=color, ha="center")
    ax_metric.text(x, 0.22, label, transform=ax_metric.transAxes,
                   fontsize=9, color=SUBTEXT, ha="center")


ax_acc = fig.add_subplot(gs[1, 0])
card_ax(ax_acc)
folds = fold_df["fold"].values
ax_acc.bar(folds, fold_df["accuracy"], color=ACCENT, alpha=0.85, label="Accuracy")
ax_acc.bar(folds, fold_df["roc_auc"], color=GREEN, alpha=0.5, label="ROC-AUC")
ax_acc.axhline(0.5, color=SUBTEXT, linestyle="--", linewidth=0.8, alpha=0.5)
ax_acc.set_title("Accuracy & ROC-AUC per Fold", color=TEXT, fontsize=11, pad=8)
ax_acc.set_xlabel("Fold", fontsize=9)
ax_acc.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax_acc.set_ylim(0, 1)


ax_pnl = fig.add_subplot(gs[1, 1])
card_ax(ax_pnl)
colors_pnl = [GREEN if v > 0 else RED for v in fold_df["total_pnl"]]
ax_pnl.bar(folds, fold_df["total_pnl"], color=colors_pnl, alpha=0.85)
cum = np.cumsum(fold_df["total_pnl"])
ax2 = ax_pnl.twinx()
ax2.plot(folds, cum, color=YELLOW, linewidth=2, marker="o", markersize=5)
ax2.set_ylabel("Cumulative PnL", color=YELLOW, fontsize=8)
ax2.tick_params(colors=YELLOW)
ax2.set_facecolor(CARD)
ax_pnl.set_title("PnL per Fold + Cumulative", color=TEXT, fontsize=11, pad=8)
ax_pnl.set_xlabel("Fold", fontsize=9)


ax_feat = fig.add_subplot(gs[1, 2])
card_ax(ax_feat)
top_n = 12
feat_top = imp_df.head(top_n)
bars = ax_feat.barh(range(top_n), feat_top["importance"].values[::-1],
                    color=ACCENT, alpha=0.85)
ax_feat.set_yticks(range(top_n))
ax_feat.set_yticklabels(feat_top["feature"].values[::-1], fontsize=8)
ax_feat.set_title("Top Feature Importances", color=TEXT, fontsize=11, pad=8)
ax_feat.set_xlabel("Importance", fontsize=9)


ax_sig = fig.add_subplot(gs[2, 0])
card_ax(ax_sig)
sig_counts = signals_df["signal"].value_counts()
colors_sig = [GREEN, RED, SUBTEXT]
wedge_labels = [f"{k}\n{v}" for k, v in sig_counts.items()]
wedges, texts, autotexts = ax_sig.pie(
    sig_counts.values, labels=wedge_labels,
    colors=[GREEN, RED, SUBTEXT][:len(sig_counts)],
    autopct="%1.0f%%", startangle=90,
    textprops={"color": TEXT, "fontsize": 9},
    wedgeprops={"edgecolor": BORDER, "linewidth": 1}
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight("bold")
ax_sig.set_title("Signal Distribution", color=TEXT, fontsize=11, pad=8)


ax_edge = fig.add_subplot(gs[2, 1])
card_ax(ax_edge)
buy_df = signals_df[signals_df["signal"] == "BUY"]
sell_df = signals_df[signals_df["signal"] == "SELL"]
ax_edge.scatter(buy_df["current_price"], buy_df["predicted_prob"],
                color=GREEN, alpha=0.7, s=40, label="BUY", zorder=3)
ax_edge.scatter(sell_df["current_price"], sell_df["predicted_prob"],
                color=RED, alpha=0.7, s=40, label="SELL", zorder=3)
ax_edge.plot([0, 1], [0, 1], color=SUBTEXT, linewidth=1, linestyle="--", alpha=0.5)
ax_edge.set_xlabel("Current Market Price", fontsize=9)
ax_edge.set_ylabel("ML Predicted Probability", fontsize=9)
ax_edge.set_title("Edge Map: Price vs ML Prediction", color=TEXT, fontsize=11, pad=8)
ax_edge.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax_edge.set_xlim(0, 1)
ax_edge.set_ylim(0, 1)


ax_winrate = fig.add_subplot(gs[2, 2])
card_ax(ax_winrate)
ax_winrate.plot(folds, fold_df["win_rate"], color=YELLOW, linewidth=2,
                marker="D", markersize=7, markerfacecolor=YELLOW)
ax_winrate.fill_between(folds, fold_df["win_rate"], alpha=0.15, color=YELLOW)
ax_winrate.axhline(0.5, color=SUBTEXT, linestyle="--", linewidth=0.8, alpha=0.5)
ax_winrate.set_title("Win Rate per Fold", color=TEXT, fontsize=11, pad=8)
ax_winrate.set_xlabel("Fold", fontsize=9)
ax_winrate.set_ylabel("Win Rate", fontsize=9)
ax_winrate.set_ylim(0, 1)

fig.text(0.5, 0.01,
         "PolySignal  |  Ensemble: GradientBoosting + RandomForest + Logistic Regression  |  29 Features  |  Walk-Forward CV",
         ha="center", fontsize=8, color=SUBTEXT)

plt.savefig("polysignal_backtest_report.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Report saved: polysignal_backtest_report.png")
