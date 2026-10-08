"""Shared restrained publication style, sized for a 5.5-inch text block."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS = {"random": "#3C5488", "adaptive": "#E64B35", "ridge": "#008B8B",
          "elastic_net": "#7E6148", "pearson": "#6B5B95", "reference": "#777777",
          "descending": "#222222", "ascending": "#AAAAAA"}
# Elastic Net uses a purple/slate tone, distinct from Pearson within diagnostic panels.
COLORS.update(elastic_net="#756BB1", pearson="#444444")
LABELS = {"random": "Random", "adaptive": "Adaptive", "ridge": "Ridge",
          "elastic_net": "Elastic Net", "marginal_correlation": "Pearson"}
MARKERS = {"random": "o", "adaptive": "^", "ridge": "o", "elastic_net": "s", "marginal_correlation": "D"}


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": .65,
        "lines.linewidth": 1.2, "lines.markersize": 3.2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#222222", "text.color": "#222222",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "jsm2026-proceedings", "savefig.dpi": 300})


def panel(ax, letter, title):
    ax.text(-.055/ax.get_position().width, 1.10, letter, transform=ax.transAxes, fontsize=10, weight="bold", va="top")
    ax.set_title(title, loc="left", pad=8)
    ax.tick_params(width=.65, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"{name}.{suffix}", dpi=300, metadata=metadata)
    plt.close(fig)
