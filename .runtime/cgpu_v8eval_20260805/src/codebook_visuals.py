"""Small W&B-ready visualizations for dictionary/codebook vector movement."""

from __future__ import annotations

from pathlib import Path

import torch


def select_codebook_vectors(vectors: torch.Tensor, max_vectors: int) -> torch.Tensor:
    """Return a stable subset of [num_vectors, dim] vectors for plotting."""
    if vectors.ndim != 2:
        raise ValueError(f"Expected [num_vectors, dim], got shape {tuple(vectors.shape)}")
    vectors = torch.nan_to_num(vectors.detach().cpu().to(torch.float32))
    count = int(vectors.size(0))
    max_vectors = max(1, int(max_vectors))
    if count <= max_vectors:
        return vectors.clone()
    indices = torch.linspace(0, count - 1, steps=max_vectors).round().to(torch.long)
    return vectors.index_select(0, indices).clone()


def _figure_to_rgb_array(fig):
    import numpy as np

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buffer.reshape(height, width, 4)[..., :3].copy()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buffer.reshape(height, width, 3).copy()


def _pca_project_snapshots(snapshots):
    import numpy as np

    arrays = [snapshot.detach().cpu().to(torch.float32).numpy() for snapshot in snapshots]
    reference = np.concatenate(arrays, axis=0)
    mean = reference.mean(axis=0, keepdims=True)
    centered = reference - mean
    try:
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2]
        total_var = float((s_vals ** 2).sum())
        pc1_var = float(s_vals[0] ** 2 / total_var * 100.0) if total_var > 0 else 0.0
        pc2_var = float(s_vals[1] ** 2 / total_var * 100.0) if len(s_vals) > 1 and total_var > 0 else 0.0
    except np.linalg.LinAlgError:
        dim = int(arrays[-1].shape[1])
        basis = np.eye(min(2, dim), dim)
        pc1_var = 0.0
        pc2_var = 0.0
    projected = [(array - mean) @ basis.T for array in arrays]
    if projected and projected[0].shape[1] == 1:
        projected = [
            np.concatenate([projection, np.zeros_like(projection)], axis=1)
            for projection in projected
        ]
    return projected, pc1_var, pc2_var


def _fixed_square_axis_limits(projected, *, margin_fraction: float = 0.08):
    import numpy as np

    all_points = np.concatenate(projected, axis=0)
    x_min = float(all_points[:, 0].min())
    x_max = float(all_points[:, 0].max())
    y_min = float(all_points[:, 1].min())
    y_max = float(all_points[:, 1].max())
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    half_span = 0.5 * span * (1.0 + 2.0 * float(max(0.0, margin_fraction)))
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def render_codebook_scatter(snapshots, steps, *, title: str):
    """Render the latest snapshot as a PCA scatter colored by movement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not snapshots:
        return None
    projected, pc1_var, pc2_var = _pca_project_snapshots(snapshots)
    x_lim, y_lim = _fixed_square_axis_limits(projected)
    latest = projected[-1]
    first = projected[0]
    disp = np.sqrt(((latest - first) ** 2).sum(axis=1)) if len(projected) > 1 else np.zeros(latest.shape[0])
    disp_norm = disp / (disp.max() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(latest[:, 0], latest[:, 1], c=disp_norm, cmap="plasma", s=18, alpha=0.85)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title} | step {int(steps[-1])}", fontsize=10)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Displacement from first snapshot", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    image = _figure_to_rgb_array(fig)
    plt.close(fig)
    return image


def save_codebook_trajectory_gif(snapshots, steps, path, *, title: str, fps: int = 2) -> Path | None:
    """Save a PCA trajectory GIF for a sequence of codebook/dictionary snapshots."""
    if len(snapshots) < 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    projected, pc1_var, pc2_var = _pca_project_snapshots(snapshots)
    x_lim, y_lim = _fixed_square_axis_limits(projected)

    final_disp = np.sqrt(((projected[-1] - projected[0]) ** 2).sum(axis=1))
    color_values = final_disp / (final_disp.max() + 1e-8)
    colors = plt.cm.plasma(color_values)
    num_vectors = int(projected[0].shape[0])

    fig, ax = plt.subplots(figsize=(9, 7))

    def update(frame_idx):
        ax.clear()
        pts = projected[frame_idx]
        for vector_idx in range(num_vectors):
            trail = np.array([projected[t][vector_idx] for t in range(frame_idx + 1)])
            ax.plot(trail[:, 0], trail[:, 1], color=colors[vector_idx], alpha=0.25, lw=0.6)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=color_values, cmap="plasma", s=18, alpha=0.85)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{title}\nstep {int(steps[frame_idx])} | frame {frame_idx + 1}/{len(projected)}",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
        if not hasattr(update, "_cbar_added"):
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("Final displacement", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            update._cbar_added = True

    anim = FuncAnimation(fig, update, frames=len(projected), interval=500)
    fig.tight_layout()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return out_path
