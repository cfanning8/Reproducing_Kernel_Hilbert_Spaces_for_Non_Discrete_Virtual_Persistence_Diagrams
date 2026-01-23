"""Figure 2: Quotient space X/A ~= S^2 v S^2 with virtual persistence diagram."""

from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman"})
FIGURE_DPI = 300


def sphere_coordinates(radius: float, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = radius * np.sin(v) * np.cos(u)
    y = radius * np.sin(v) * np.sin(u)
    z = radius * np.cos(v)
    return x, y, z


def set_axes_equal_3d(ax: plt.Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    x_limits = (float(np.min(X)), float(np.max(X)))
    y_limits = (float(np.min(Y)), float(np.max(Y)))
    z_limits = (float(np.min(Z)), float(np.max(Z)))
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)
    ax.set_box_aspect((1, 1, 1))


def render(output_path: Path) -> None:
    """Generate Figure 2: Two 2-spheres touching at point [A] with signed multiset points."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#ffffff")

    radius = 1.0
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)

    X1, Y1, Z1 = sphere_coordinates(radius, U, V)
    X1 = X1 - radius
    X2, Y2, Z2 = sphere_coordinates(radius, U, V)
    X2 = X2 + radius

    wireframe_kwargs = {
        "rstride": 8,
        "cstride": 8,
        "color": "#9a9a9a",
        "linewidth": 0.8,
        "alpha": 0.45,
    }

    ls = LightSource(azdeg=120, altdeg=55)

    def sphere_normals(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx = np.sin(v) * np.cos(u)
        ny = np.sin(v) * np.sin(u)
        nz = np.cos(v)
        return nx, ny, nz

    nx1, ny1, nz1 = sphere_normals(U, V)
    normals1 = np.stack([nx1, ny1, nz1], axis=-1)
    shaded1 = ls.shade_normals(normals1)
    facecolors1 = plt.get_cmap("Greys")(shaded1)

    nx2, ny2, nz2 = sphere_normals(U, V)
    normals2 = np.stack([nx2, ny2, nz2], axis=-1)
    shaded2 = ls.shade_normals(normals2)
    facecolors2 = plt.get_cmap("Greys")(shaded2)

    ax.plot_surface(X1, Y1, Z1, facecolors=facecolors1, alpha=0.12, rstride=4, cstride=4, linewidth=0)
    ax.plot_wireframe(X1, Y1, Z1, **wireframe_kwargs)
    ax.plot_surface(X2, Y2, Z2, facecolors=facecolors2, alpha=0.12, rstride=4, cstride=4, linewidth=0)
    ax.plot_wireframe(X2, Y2, Z2, **wireframe_kwargs)

    ax.scatter([0], [0], [0], color="#c0392b", s=150, zorder=10, edgecolors="#000000", linewidths=1.5)
    ax.text(0, 0, 0.2, "[A]", fontsize=12, fontweight="bold", color="#c0392b", ha="center")

    def point_on_sphere(center_x: float, u: float, v: float, r: float) -> Tuple[float, float, float]:
        x = center_x + r * np.sin(v) * np.cos(u)
        y = r * np.sin(v) * np.sin(u)
        z = r * np.cos(v)
        return (x, y, z)

    pos_points = [
        point_on_sphere(-radius, 0.8, 1.2, radius),
        point_on_sphere(-radius, 2.5, 0.8, radius),
        point_on_sphere(-radius, 4.0, 1.5, radius),
    ]
    pos_weights = [1, -2, 1]
    neg_points = [
        point_on_sphere(radius, 0.5, 1.8, radius),
        point_on_sphere(radius, 2.2, 0.6, radius),
        point_on_sphere(radius, 3.8, 1.3, radius),
    ]
    neg_weights = [-1, 2, -1]
    all_points = pos_points + neg_points
    all_weights = pos_weights + neg_weights

    for (px, py, pz), weight in zip(all_points, all_weights):
        size = 100 + abs(weight) * 30
        ax.scatter([px], [py], [pz], color="#2c6df2", s=size, zorder=8,
                  edgecolors="#000000", linewidths=1.5, alpha=0.9)
        label = f"+{weight}" if weight > 0 else str(weight)
        ax.text(px, py, pz + 0.15, label, fontsize=9, color="#2c6df2",
               fontweight="bold", ha="center")

    ax.text(-radius, 0, -1.5, "Inside region", fontsize=11,
           color="#000000", ha="center", va="top",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(radius, 0, -1.5, "Outside region", fontsize=11,
           color="#000000", ha="center", va="top",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8))
    all_X = np.concatenate([X1.flatten(), X2.flatten()])
    all_Y = np.concatenate([Y1.flatten(), Y2.flatten()])
    all_Z = np.concatenate([Z1.flatten(), Z2.flatten()])
    set_axes_equal_3d(ax, all_X, all_Y, all_Z)

    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, transparent=True, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    render(output_dir / "figure_2_quotient_virtual_diagram.png")
