# src/plottingutils.py
"""
Publication-ready plotting helpers.
All functions save a PDF to `path` and also return the Figure object.
No watermarks. No external branding.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "legend.framealpha": 0.7,
})

_PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706",
            "#7C3AED", "#0891B2", "#DB2777"]


def _save(fig: plt.Figure, path: str) -> plt.Figure:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches="tight", facecolor="white")
    return fig


# ── 1. Line plot ──────────────────────────────────────────────────────────────
def line_plot(df: pd.DataFrame,
              x: str,
              y: str,
              hue: str,
              title: str = "",
              xlabel: str = None,
              ylabel: str = None,
              path: str = None) -> plt.Figure:
    """Multi-line chart with confidence-style markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups  = df[hue].unique()

    for i, grp in enumerate(groups):
        sub = df[df[hue] == grp].sort_values(x)
        ax.plot(sub[x], sub[y],
                marker="o", linewidth=2.2, markersize=6,
                color=_PALETTE[i % len(_PALETTE)], label=str(grp))

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend(title=hue, loc="lower right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, path)


# ── 2. Scatter plot ───────────────────────────────────────────────────────────
def scatter_plot(df: pd.DataFrame,
                 x: str,
                 y: str,
                 label_col: str = None,
                 title: str = "",
                 xlabel: str = None,
                 ylabel: str = None,
                 path: str = None) -> plt.Figure:
    """Scatter plot; optionally colour-codes by label_col."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if label_col and label_col in df.columns:
        labels = df[label_col].unique()
        for i, lbl in enumerate(labels):
            sub = df[df[label_col] == lbl]
            ax.scatter(sub[x], sub[y],
                       color=_PALETTE[i % len(_PALETTE)],
                       s=120, label=str(lbl), alpha=0.85, edgecolors="white")
        ax.legend(title=label_col)
    else:
        ax.scatter(df[x], df[y],
                   color=_PALETTE[0], s=120, alpha=0.85, edgecolors="white")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    return _save(fig, path)


# ── 3. Heatmap ────────────────────────────────────────────────────────────────
def heatmap(df: pd.DataFrame,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            fmt: str = ".2f",
            path: str = None) -> plt.Figure:
    """Annotated heatmap from a pivot-style DataFrame."""
    import matplotlib.colors as mcolors
    fig, ax  = plt.subplots(figsize=(max(6, df.shape[1]),
                                      max(4, df.shape[0])))
    data     = df.values.astype(float)
    im       = ax.imshow(data, cmap="Blues", aspect="auto",
                         vmin=np.nanmin(data), vmax=np.nanmax(data))

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=10)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if not np.isnan(val):
                ax.text(c, r, format(val, fmt),
                        ha="center", va="center",
                        color="white" if val > np.nanmean(data) else "black",
                        fontsize=9)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    return _save(fig, path)


# ── 4. Bar plot ───────────────────────────────────────────────────────────────
def bar_plot(df: pd.DataFrame,
             x: str,
             y: str,
             hue: str = None,
             title: str = "",
             xlabel: str = None,
             ylabel: str = None,
             path: str = None) -> plt.Figure:
    """Grouped or simple bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if hue and hue in df.columns:
        groups  = df[hue].unique()
        n       = len(groups)
        x_vals  = df[x].unique()
        x_pos   = np.arange(len(x_vals))
        width   = 0.8 / n

        for i, grp in enumerate(groups):
            sub    = df[df[hue] == grp].set_index(x)
            heights = [float(sub.loc[xv, y]) if xv in sub.index else 0
                       for xv in x_vals]
            ax.bar(x_pos + i * width, heights, width,
                   label=str(grp),
                   color=_PALETTE[i % len(_PALETTE)], alpha=0.85)

        ax.set_xticks(x_pos + width * (n - 1) / 2)
        ax.set_xticklabels(x_vals, rotation=30, ha="right")
        ax.legend(title=hue)
    else:
        ax.bar(df[x].astype(str), df[y],
               color=_PALETTE[0], alpha=0.85)
        ax.set_xticklabels(df[x].astype(str), rotation=30, ha="right")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    return _save(fig, path)

# aliases to match notebook import names
lineplot    = line_plot
scatterplot = scatter_plot
barplot     = bar_plot