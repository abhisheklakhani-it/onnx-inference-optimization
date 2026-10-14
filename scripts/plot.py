"""Charts from results/benchmark.json -> docs/throughput.png and docs/size.png.

    python scripts/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onnxopt.paths import DISPLAY, DOCS, RESULTS, VARIANTS  # noqa: E402

BLUE = "#2a78d6"
BATCH_RAMP = {"1": "#86b6ef", "8": "#2a78d6", "32": "#104281"}  # ordinal: light -> dark = small -> large batch
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style_axis(ax, grid: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0, colors=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def throughput_chart(summary: dict) -> None:
    runs = summary["variants"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axis(ax)
    names = list(reversed(VARIANTS))  # PyTorch at the top
    height = 0.26
    for j, (batch, color) in enumerate(BATCH_RAMP.items()):
        ys = [i + (1 - j) * (height + 0.02) for i in range(len(names))]
        values = [runs[n]["batch_sizes"][batch]["articles_per_s"] for n in names]
        ax.barh(ys, values, height=height, color=color, label=f"batch {batch}", zorder=2)
        if batch == "1":
            for y, v in zip(ys, values):
                ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(names)), [DISPLAY[n] for n in names], color=INK)
    ax.set_xlabel(f"articles per second ({summary['timing_articles']} test articles x {summary['timing_rounds']} rounds, "
                  f"{summary['machine']['threads']} threads, {summary['machine']['cpu']})", color=MUTED)
    ax.set_title("Throughput by variant and batch size", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(DOCS / "throughput.png", dpi=160)


def size_chart(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURFACE)
    style_axis(ax)
    names = list(reversed(VARIANTS))
    sizes = [runs[n]["size_mb"] for n in names]
    ax.barh(range(len(names)), sizes, height=0.55, color=BLUE, zorder=2)
    for i, s in enumerate(sizes):
        ax.text(s + 3, i, f"{s:.0f} MB", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(names)), [DISPLAY[n] for n in names], color=INK)
    ax.set_xlim(0, max(sizes) * 1.15)
    ax.set_title("Model file size", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "size.png", dpi=160)


def main() -> None:
    summary = json.loads((RESULTS / "benchmark.json").read_text())
    DOCS.mkdir(exist_ok=True)
    throughput_chart(summary)
    size_chart(summary["variants"])
    print(f"wrote {DOCS / 'throughput.png'} and {DOCS / 'size.png'}")


if __name__ == "__main__":
    main()
