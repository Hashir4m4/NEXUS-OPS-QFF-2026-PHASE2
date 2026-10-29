"""
NEXUS-OPS | QFF 2026 Phase 2
Figure Generation Script for Visual Analysis and Report Integration
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_all_figures():
    print("Generating figures for reports and notebook...")

    # 1. Loss Curve
    ideal_file = METRICS_DIR / "vqc_4qubit_ideal.json"
    if ideal_file.exists():
        with open(ideal_file) as f:
            ideal = json.load(f)
        losses = ideal.get("epoch_losses", [])
        if losses:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, marker="o", color="#1f77b4", linewidth=2.5, markersize=6, label="Training MSE Loss")
            ax.set_title("NEXUS-OPS PS-Q1: 4-Qubit VQC Optimization Trajectory", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
            ax.set_xticks(list(epochs))
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(frameon=True)
            loss_fig_path = FIG_DIR / "vqc_training_loss.png"
            fig.savefig(loss_fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved: {loss_fig_path.name}")

    # 2. Confusion Matrices (Ideal vs Noisy)
    noisy_file = METRICS_DIR / "vqc_4qubit_noisy.json"
    if ideal_file.exists() and noisy_file.exists():
        with open(ideal_file) as f:
            ideal = json.load(f)
        with open(noisy_file) as f:
            noisy = json.load(f)

        cm_ideal = np.array(ideal["confusion_matrix"])
        cm_noisy = np.array(noisy["confusion_matrix"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        # Ideal CM
        im1 = ax1.imshow(cm_ideal, cmap="Blues", vmin=0, vmax=900)
        ax1.set_title(f"Ideal Simulation (Macro-F1: {ideal['macro_f1']:.4f})", fontsize=12, fontweight="bold")
        ax1.set_xticks([0, 1], ["Normal", "Arrhythmia"])
        ax1.set_yticks([0, 1], ["Normal", "Arrhythmia"])
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")
        for i in range(2):
            for j in range(2):
                val = cm_ideal[i, j]
                color = "white" if val > 450 else "black"
                ax1.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

        # Noisy CM
        im2 = ax2.imshow(cm_noisy, cmap="Blues", vmin=0, vmax=900)
        ax2.set_title(f"Noisy QFF2026-v2 (Macro-F1: {noisy['macro_f1']:.4f})", fontsize=12, fontweight="bold")
        ax2.set_xticks([0, 1], ["Normal", "Arrhythmia"])
        ax2.set_yticks([0, 1], ["Normal", "Arrhythmia"])
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        for i in range(2):
            for j in range(2):
                val = cm_noisy[i, j]
                color = "white" if val > 450 else "black"
                ax2.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

        fig.suptitle("Validation Confusion Matrix: Ideal vs Benchmark Noise (8192 shots)", fontsize=14, fontweight="bold", y=1.02)
        cm_fig_path = FIG_DIR / "confusion_matrix_comparison.png"
        fig.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {cm_fig_path.name}")

    # 3. Ablation Study Chart
    abl_file = METRICS_DIR / "ablation_study.json"
    if abl_file.exists():
        with open(abl_file) as f:
            abl = json.load(f)

        names = list(abl.keys())
        f1_scores = [abl[k]["macro_f1"] for k in names]
        times = [abl[k]["training_time_seconds"] for k in names]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(names))
        width = 0.38

        color1 = "#2ca02c"
        color2 = "#ff7f0e"

        rects1 = ax1.bar(x - width/2, f1_scores, width, label="Macro-F1", color=color1, alpha=0.85)
        ax1.set_ylabel("Validation Macro-F1", color=color1, fontsize=12)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim([0.4, 1.05])
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=15, ha="right", fontsize=9.5)

        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, times, width, label="Train Time (s)", color=color2, alpha=0.85)
        ax2.set_ylabel("Training Time (seconds)", color=color2, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.grid(False)

        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.suptitle("Architecture Ablation Study: Performance vs Compute Cost", fontsize=13, fontweight="bold")
        fig.tight_layout()
        abl_fig_path = FIG_DIR / "ablation_study_chart.png"
        fig.savefig(abl_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {abl_fig_path.name}")

    print("All figures successfully generated.")


if __name__ == "__main__":
    generate_all_figures()

