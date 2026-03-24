from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

"""
绘制任务分组准确率柱状图。
用法：
  cd D:\Code\DeepLearning\CelebA\results
  python plot_fsc_eval.py
"""

CSV_PATH = Path(__file__).parent / "eval_summary.csv"
OUT_DIR = Path(__file__).parent

# Single-task breakdown figures: vertical size (bar pixel height ~ 2/3 of old 7in)
SINGLE_TASK_FIGSIZE = (11.0, 7.0 * 2.0 / 3.0)

METHOD_ORDER = [
    "Baseline (ERM)",
    "FSC (Unbalanced)",
    "FSC (Oversampling)",
    "FSC (Reweighting)",
]

TASK_TITLES = {
    "BlondHair_Male": "Blond × Male - Accuracy Breakdown by Demographic Groups",
    "MouthOpen_Smiling": "Mouth × Smiling - Accuracy Breakdown by Demographic Groups",
}


def method_xtick_label(label: str) -> str:
    s = str(label).strip()
    if not s.startswith("FSC"):
        return s
    return s + "\n" + r"$\mathbf{Ours}$"


def method_xtick_labels(labels) -> list[str]:
    return [method_xtick_label(x) for x in labels]


def style_method_xticklabels(ax: plt.Axes) -> None:
    for lbl in ax.get_xticklabels():
        lbl.set_multialignment("center")
        lbl.set_ha("center")


def _safe_task_name(task: str) -> str:
    return task.lower().replace(" ", "_")


def _task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["task"] == task].copy()
    task_df["method_label"] = pd.Categorical(task_df["method_label"], METHOD_ORDER, ordered=True)
    task_df = task_df.sort_values("method_label")
    if task_df.empty:
        raise ValueError(f"No rows found for task: {task}")
    return task_df


def plot_task_axes(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    *,
    title: str | None = None,
    show_legend: bool = True,
    legend_fontsize: int = 9,
) -> None:
    task_df = _task_frame(df, task)
    group_names = [task_df.iloc[0][f"group{i}_name"] for i in range(4)]
    overall = task_df["overall_acc"].to_numpy()
    g0 = task_df["acc_g0"].to_numpy()
    g1 = task_df["acc_g1"].to_numpy()
    g2 = task_df["acc_g2"].to_numpy()
    g3 = task_df["acc_g3"].to_numpy()

    x = np.arange(len(task_df))
    width = 0.16

    ax.bar(x - 2 * width, overall, width, color="#34495e", edgecolor="black", label="Overall Accuracy")
    ax.bar(x - width, g0, width, color="#5dade2", edgecolor="black", label=group_names[0])
    ax.bar(x, g1, width, color="#58d68d", edgecolor="black", label=group_names[1])
    ax.bar(x + width, g2, width, color="#ec7063", edgecolor="black", label=group_names[2])
    ax.bar(x + 2 * width, g3, width, color="#d4ac0d", edgecolor="black", label=group_names[3])

    # Per-method: max/min among the four demographic bars + gap (percentage points)
    group_offsets = np.array([-width, 0.0, width, 2 * width])
    label_dy = 0.012
    line_dy = 0.008
    for i in range(len(x)):
        heights = np.array([g0[i], g1[i], g2[i], g3[i]], dtype=float)
        centers = x[i] + group_offsets
        imax = int(np.argmax(heights))
        imin = int(np.argmin(heights))
        if imax == imin:
            continue
        h_max, h_min = heights[imax], heights[imin]
        cx_max, cx_min = centers[imax], centers[imin]
        ax.text(
            cx_max,
            h_max + label_dy,
            f"{100 * h_max:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # Min group: put label inside the bar (lower part), not above the top edge
        min_label_threshold = 0.11
        if h_min >= min_label_threshold:
            ax.text(
                cx_min,
                h_min * 0.98,
                f"{100 * h_min:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="0.15",
            )
        else:
            ax.text(
                cx_min,
                h_min + label_dy,
                f"{100 * h_min:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        y0 = h_max + line_dy
        y1 = h_min + line_dy
        gap_pp = 100.0 * (h_max - h_min)
        arr = FancyArrowPatch(
            (cx_max, y0),
            (cx_min, y1),
            arrowstyle="<->",
            mutation_scale=8,
            color="black",
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
        ax.add_patch(arr)
        mid_x = (cx_max + cx_min) / 2
        mid_y = (y0 + y1) / 2
        ax.text(
            mid_x,
            mid_y,
            f"gap {gap_pp:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#f5e6d3",
                edgecolor="0.45",
                linewidth=0.6,
            ),
            zorder=6,
        )

    baseline_row = task_df[task_df["method_label"] == "Baseline (ERM)"].iloc[0]
    baseline_worst = float(baseline_row["wga"])
    ax.axhline(
        y=baseline_worst,
        color="#9b6d6d",
        linestyle="--",
        linewidth=1.4,
        alpha=0.65,
        label="Baseline Worst Group",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(method_xtick_labels(task_df["method_label"]), fontsize=9, rotation=10)
    style_method_xticklabels(ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlabel("De-biasing Method", fontsize=11)
    ax.set_title(title or TASK_TITLES.get(task, task), fontsize=13, weight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
            fontsize=legend_fontsize,
        )


def plot_task(df: pd.DataFrame, task: str) -> Path:
    fig, ax = plt.subplots(figsize=SINGLE_TASK_FIGSIZE)
    plot_task_axes(ax, df, task, title=TASK_TITLES.get(task, task), show_legend=True, legend_fontsize=9)
    fig.tight_layout(rect=[0.02, 0.02, 0.74, 0.98])
    out_path = OUT_DIR / f"fsc_eval_{_safe_task_name(task)}_acc.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    outputs = [
        plot_task(df, "BlondHair_Male"),
        plot_task(df, "MouthOpen_Smiling"),
    ]
    for out in outputs:
        print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
