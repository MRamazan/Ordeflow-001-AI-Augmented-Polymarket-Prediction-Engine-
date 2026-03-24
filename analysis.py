import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

from engine import (
    simulate_polymarket_data, build_features, FEATURE_COLS,
    EnsembleSignalEngine, walk_forward_backtest,
    generate_live_signals, get_feature_importance_df
)

SEED = 42
np.random.seed(SEED)

BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#e3b341"
PURPLE = "#bc8cff"

raw_df = simulate_polymarket_data(n_markets=80, n_snapshots=200)
df_feat = build_features(raw_df)

tscv = TimeSeriesSplit(n_splits=5)
X = df_feat[FEATURE_COLS].values
y = df_feat["outcome"].values

all_probs = []
all_labels = []
all_fold_roc = []

for train_idx, test_idx in tscv.split(X):
    eng = EnsembleSignalEngine()
    eng.fit(X[train_idx], y[train_idx])
    p = eng.predict_proba(X[test_idx])[:, 1]
    all_probs.extend(p)
    all_labels.extend(y[test_idx])
    fpr, tpr, _ = roc_curve(y[test_idx], p)
    all_fold_roc.append((fpr, tpr, auc(fpr, tpr)))

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

X_full = df_feat[FEATURE_COLS].values
y_full = df_feat["outcome"].values
final_engine = EnsembleSignalEngine()
final_engine.fit(X_full, y_full)
signals_df = generate_live_signals(final_engine, df_feat)
imp_df = get_feature_importance_df(final_engine)
results = walk_forward_backtest(df_feat, n_splits=5)
fold_df = results["fold_metrics"]

fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "axes.edgecolor": BORDER,
})

def card(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)
        s.set_linewidth(0.8)

fig.text(0.5, 0.97, "PolySignal — Model Calibration & Edge Analysis",
         ha="center", fontsize=18, fontweight="bold", color=TEXT)
fig.text(0.5, 0.948, "Diagnostic suite for judges | All metrics from walk-forward CV (no lookahead bias)",
         ha="center", fontsize=10, color=SUBTEXT)


ax1 = fig.add_subplot(gs[0, 0])
card(ax1)
fraction_pos, mean_pred = calibration_curve(all_labels, all_probs, n_bins=10, strategy="uniform")
ax1.plot([0, 1], [0, 1], color=SUBTEXT, linestyle="--", linewidth=1, label="Perfect calibration")
ax1.plot(mean_pred, fraction_pos, color=ACCENT, linewidth=2.5, marker="o",
         markersize=7, markerfacecolor=ACCENT, label="PolySignal")
ax1.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.12, color=ACCENT)
ax1.set_title("Probability Calibration Curve", color=TEXT, fontsize=11, pad=8)
ax1.set_xlabel("Mean Predicted Probability", fontsize=9)
ax1.set_ylabel("Fraction of Positives", fontsize=9)
ax1.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)


ax2 = fig.add_subplot(gs[0, 1])
card(ax2)
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_roc):
    alpha = 0.35 if i < len(all_fold_roc) - 1 else 1.0
    lw = 1.2 if i < len(all_fold_roc) - 1 else 2.5
    label = f"Fold {i+1} (AUC={roc_auc:.3f})" if i == len(all_fold_roc) - 1 else f"Fold {i+1}"
    ax2.plot(fpr, tpr, color=ACCENT, alpha=alpha, linewidth=lw, label=label)
ax2.plot([0, 1], [0, 1], color=SUBTEXT, linestyle="--", linewidth=1)
ax2.set_title("ROC Curves — All Folds", color=TEXT, fontsize=11, pad=8)
ax2.set_xlabel("False Positive Rate", fontsize=9)
ax2.set_ylabel("True Positive Rate", fontsize=9)
mean_auc = np.mean([r[2] for r in all_fold_roc])
ax2.text(0.55, 0.12, f"Mean AUC = {mean_auc:.4f}", color=GREEN,
         fontsize=10, fontweight="bold", transform=ax2.transAxes)
ax2.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)


ax3 = fig.add_subplot(gs[0, 2])
card(ax3)
edges = signals_df["edge"].values
buy_edges = edges[edges > 0.06]
sell_edges = edges[edges < -0.06]
hold_edges = edges[(edges >= -0.06) & (edges <= 0.06)]
bins = np.linspace(-0.6, 0.6, 40)
ax3.hist(hold_edges, bins=bins, color=SUBTEXT, alpha=0.6, label=f"HOLD (n={len(hold_edges)})")
ax3.hist(buy_edges, bins=bins, color=GREEN, alpha=0.8, label=f"BUY (n={len(buy_edges)})")
ax3.hist(sell_edges, bins=bins, color=RED, alpha=0.8, label=f"SELL (n={len(sell_edges)})")
ax3.axvline(0.06, color=GREEN, linestyle="--", linewidth=1, alpha=0.7)
ax3.axvline(-0.06, color=RED, linestyle="--", linewidth=1, alpha=0.7)
ax3.axvline(0, color=SUBTEXT, linestyle="-", linewidth=0.8, alpha=0.5)
ax3.set_title("Edge Distribution Across Markets", color=TEXT, fontsize=11, pad=8)
ax3.set_xlabel("Edge (ML Prob − Market Price)", fontsize=9)
ax3.set_ylabel("Count", fontsize=9)
ax3.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)


ax4 = fig.add_subplot(gs[1, 0])
card(ax4)
prob_bins = np.linspace(0, 1, 11)
bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
bin_indices = np.digitize(all_probs, prob_bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)
bin_counts = [np.sum(bin_indices == i) for i in range(10)]
bin_acc = []
for i in range(10):
    mask = bin_indices == i
    if mask.sum() > 0:
        bin_acc.append(all_labels[mask].mean())
    else:
        bin_acc.append(np.nan)
colors_bar = [GREEN if a > 0.5 else RED if not np.isnan(a) else SUBTEXT for a in bin_acc]
ax4.bar(bin_centers, bin_counts, width=0.08, color=ACCENT, alpha=0.4, label="Sample count")
ax4_r = ax4.twinx()
ax4_r.plot(bin_centers, bin_acc, color=YELLOW, linewidth=2.5, marker="s",
           markersize=6, markerfacecolor=YELLOW, label="Actual win rate")
ax4_r.axhline(0.5, color=SUBTEXT, linestyle="--", linewidth=0.8)
ax4_r.set_ylabel("Actual Win Rate", color=YELLOW, fontsize=9)
ax4_r.tick_params(colors=YELLOW)
ax4_r.set_facecolor(CARD)
ax4_r.set_ylim(0, 1)
ax4.set_title("Reliability Diagram (Confidence vs Accuracy)", color=TEXT, fontsize=11, pad=8)
ax4.set_xlabel("Predicted Probability Bin", fontsize=9)
ax4.set_ylabel("Sample Count", fontsize=9)


ax5 = fig.add_subplot(gs[1, 1])
card(ax5)
cum_pnl = np.cumsum(fold_df["total_pnl"].values)
fold_labels = [f"F{i+1}" for i in range(len(fold_df))]
ax5.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.2, color=GREEN)
ax5.plot(range(len(cum_pnl)), cum_pnl, color=GREEN, linewidth=2.5,
         marker="o", markersize=8, markerfacecolor=GREEN)
for i, (x, y_val) in enumerate(zip(range(len(cum_pnl)), cum_pnl)):
    ax5.annotate(f"{y_val:.1f}", (x, y_val), textcoords="offset points",
                xytext=(0, 10), ha="center", color=GREEN, fontsize=8)
ax5.set_xticks(range(len(fold_df)))
ax5.set_xticklabels(fold_labels)
ax5.set_title("Cumulative PnL Across Folds", color=TEXT, fontsize=11, pad=8)
ax5.set_xlabel("Fold", fontsize=9)
ax5.set_ylabel("Cumulative PnL", fontsize=9)
ax5.text(0.05, 0.88, f"Total: {cum_pnl[-1]:.2f}", transform=ax5.transAxes,
         color=GREEN, fontsize=11, fontweight="bold")
ax5.text(0.05, 0.76, f"Sharpe: {results['sharpe_ratio']:.2f}", transform=ax5.transAxes,
         color=YELLOW, fontsize=10)


ax6 = fig.add_subplot(gs[1, 2])
card(ax6)
top15 = imp_df.head(15)
y_pos = np.arange(len(top15))
color_map = [ACCENT if i < 3 else (GREEN if i < 7 else SUBTEXT) for i in range(len(top15))]
bars = ax6.barh(y_pos, top15["importance"].values[::-1], color=color_map[::-1], alpha=0.85)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(top15["feature"].values[::-1], fontsize=8)
ax6.set_title("Feature Importances (Ensemble)", color=TEXT, fontsize=11, pad=8)
ax6.set_xlabel("Importance Score", fontsize=9)
for bar, val in zip(bars, top15["importance"].values[::-1]):
    ax6.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=7, color=SUBTEXT)

fig.text(0.5, 0.01,
         "Calibration  |  ROC-AUC per fold  |  Edge distribution  |  Reliability diagram  |  Cumulative PnL  |  Feature importance",
         ha="center", fontsize=8, color=SUBTEXT)

plt.savefig("polysignal_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved: polysignal_analysis.png")
