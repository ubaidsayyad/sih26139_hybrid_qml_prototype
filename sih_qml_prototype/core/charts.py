"""
charts.py
---------
Generates the comparison / explainability / training-curve PNGs shown on
the dashboard, using matplotlib (Agg backend, no display needed), styled
for the dark-mode quantum research console UI.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Dark theme palette
BG = "#0B1120"
CARD_BG = "#0D1526"
GRID = "#1E293B"
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

PRIMARY = "#8B5CF6"       # Electric Violet
QUANTUM = "#00E5FF"       # Electric Cyan
QUANTUM_ALT = "#00F5D4"   # Soft Quantum Teal Glow
CLASSICAL = "#475569"     # Deep Slate
CLASSICAL_LIGHT = "#94A3B8"

def _style(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    if title:
        ax.set_title(title, fontsize=11, weight="bold", color=TEXT_MAIN, pad=12, family="sans-serif")


def metric_comparison_chart(results, out_path):
    labels = ["LogReg\n(30 feat)", "RandomForest\n(30 feat)", "LogReg\n(4 PCA feat)", "Hybrid VQC\n(4 qubits)"]
    keys = ["logistic_regression", "random_forest", "logistic_regression_same_4_features", "hybrid_vqc"]
    accs = [results[k]["accuracy"] * 100 for k in keys]
    aucs = [results[k]["auc"] * 100 for k in keys]
    colors = [CLASSICAL, CLASSICAL, CLASSICAL, QUANTUM]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=140, facecolor=BG)
    x = np.arange(len(labels))
    w = 0.32
    b1 = ax.bar(x - w / 2, accs, width=w, label="Accuracy", color=colors, alpha=0.9, edgecolor="none")
    b2 = ax.bar(x + w / 2, aucs, width=w, label="AUC", color=colors, alpha=0.45, edgecolor="none")
    
    # Highlight quantum bars with glowing borders
    b1[3].set_edgecolor(QUANTUM_ALT)
    b1[3].set_linewidth(1.5)
    b2[3].set_edgecolor(QUANTUM)
    b2[3].set_linewidth(1.2)

    for i, (a, u) in enumerate(zip(accs, aucs)):
        c = QUANTUM if i == 3 else TEXT_MAIN
        ax.text(i - w / 2, a + 1.2, f"{a:.1f}%", ha="center", fontsize=8, color=c, weight="semibold")
        ax.text(i + w / 2, u + 1.2, f"{u:.1f}%", ha="center", fontsize=8, color=c, alpha=0.9)
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=TEXT_MAIN)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)", fontsize=9)
    _style(ax, "Classical Baselines vs Hybrid Quantum Model (Held-out Test Set)")
    
    leg = ax.legend(frameon=True, facecolor=CARD_BG, edgecolor=GRID, loc="lower right", labelcolor=TEXT_MAIN, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def training_loss_chart(loss_history, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=140, facecolor=BG)
    epochs_range = range(1, len(loss_history) + 1)
    ax.plot(epochs_range, loss_history, color=QUANTUM, linewidth=2.2, label="BCE Loss")
    ax.fill_between(epochs_range, loss_history, alpha=0.15, color=QUANTUM)
    
    # Add subtle point markers
    if len(loss_history) <= 30:
        ax.scatter(epochs_range, loss_history, color=QUANTUM_ALT, s=18, zorder=5)
        
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=9)
    _style(ax, "Quantum Circuit Convergence (Parameter-Shift Gradients)")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def importance_chart(importances, title, color, out_path):
    feats = [d["feature"] for d in importances]
    vals = [d["importance"] for d in importances]
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=140, facecolor=BG)
    bars = ax.barh(feats, vals, color=color, alpha=0.85, height=0.55)
    
    # Add subtle border to bars
    for bar in bars:
        bar.set_edgecolor(color)
        bar.set_linewidth(1.0)
        
    ax.invert_yaxis()
    ax.set_xlabel("Importance (ΔAUC when feature is shuffled)", fontsize=8.5)
    _style(ax, title)
    ax.tick_params(colors=TEXT_MAIN, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def shap_importance_chart(importances, title, color, out_path):
    feats = [d["feature"] for d in importances]
    vals = [d["importance"] for d in importances]
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=140, facecolor=BG)
    bars = ax.barh(feats, vals, color=color, alpha=0.85, height=0.55)
    
    for bar in bars:
        bar.set_edgecolor(color)
        bar.set_linewidth(1.0)
        
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (Feature Attribution Magnitude)", fontsize=8.5)
    _style(ax, title)
    ax.tick_params(colors=TEXT_MAIN, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def roc_chart(y_test, proba_dict, out_path):
    """proba_dict: {label: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140, facecolor=BG)
    colors = {"Hybrid VQC": QUANTUM, "Logistic Regression": PRIMARY}
    
    for label, (yt, yp) in proba_dict.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        c = colors.get(label, QUANTUM_ALT)
        ax.plot(fpr, tpr, label=label, linewidth=2.2, color=c)
        
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1.2, label="Chance level (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    _style(ax, "Receiver Operating Characteristic (ROC)")
    ax.legend(frameon=True, facecolor=CARD_BG, edgecolor=GRID, loc="lower right", labelcolor=TEXT_MAIN, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
