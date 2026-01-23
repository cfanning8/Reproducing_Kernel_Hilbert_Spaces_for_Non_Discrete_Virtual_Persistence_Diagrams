"""Figure 1: Ambient metric pair (C, |.|, S^1)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman"})
FIGURE_DPI = 300


def render(output_path: Path) -> None:
    """Generate Figure 1: Complex plane with S^1 and strengthened metric."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=FIGURE_DPI)
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")

    ax.axhline(y=0, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.5)

    theta_circle = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta_circle)
    circle_y = np.sin(theta_circle)
    ax.plot(circle_x, circle_y, color="#c0392b", linewidth=2.5, zorder=3)

    x_complex = 0.4 + 0.3j
    y_complex = 1.8 + 0.6j
    x_re, x_im = x_complex.real, x_complex.imag
    y_re, y_im = y_complex.real, y_complex.imag

    ax.scatter([x_re], [x_im], color="#2c6df2", s=120, zorder=5, edgecolors="#000000", linewidths=1.5)
    ax.scatter([y_re], [y_im], color="#2c6df2", s=120, zorder=5, edgecolors="#000000", linewidths=1.5)
    ax.plot([x_re, y_re], [x_im, y_im], color="#2c6df2", linewidth=2.0, linestyle="-", alpha=0.7, zorder=2)

    x_norm = abs(x_complex)
    x_nearest = x_complex / x_norm if x_norm > 0 else 1.0
    x_nearest_re, x_nearest_im = x_nearest.real, x_nearest.imag
    y_norm = abs(y_complex)
    y_nearest = y_complex / y_norm if y_norm > 0 else 1.0
    y_nearest_re, y_nearest_im = y_nearest.real, y_nearest.imag

    ax.plot([x_re, x_nearest_re], [x_im, x_nearest_im], color="#2c6df2", linewidth=2.0,
            linestyle="--", alpha=0.7, zorder=2)
    ax.plot([y_re, y_nearest_re], [y_im, y_nearest_im], color="#2c6df2", linewidth=2.0,
            linestyle="--", alpha=0.7, zorder=2)

    ax.text(0.0, 2.0, r"$\mathbb{C}$", fontsize=32, color="#000000", ha="center", va="top")
    ax.text(0.0, 1.15, r"$S^1$", fontsize=14, color="#c0392b", ha="center", va="bottom")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("Re(z)", fontsize=12)
    ax.set_ylabel("Im(z)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    render(output_dir / "figure_1_metric_pair.png")
