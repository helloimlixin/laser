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


def render_dictionary_usage_scatter(
    atoms: torch.Tensor,
    usage: torch.Tensor,
    contribution: torch.Tensor,
    *,
    atom_ids: torch.Tensor | None = None,
    step: int = 0,
    title: str = "Dictionary atoms",
):
    """Render dictionary geometry together with empirical sparse-code usage.

    PCA coordinates provide a geometry for the atoms; raw atom IDs do not. The
    left panel therefore shows the projected dictionary with validation-active
    atoms colored by selection count and sized by absolute coefficient
    contribution. The right panel makes load concentration explicit by plotting
    selection count against mean absolute coefficient for each active atom.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    atoms = torch.nan_to_num(atoms.detach().cpu().to(torch.float32))
    usage = torch.nan_to_num(usage.detach().cpu().to(torch.float32)).reshape(-1)
    contribution = torch.nan_to_num(
        contribution.detach().cpu().to(torch.float32)
    ).reshape(-1)
    if atoms.ndim != 2:
        raise ValueError(f"Expected atoms [N,D], got {tuple(atoms.shape)}")
    count = int(atoms.size(0))
    if usage.numel() != count or contribution.numel() != count:
        raise ValueError(
            "atoms, usage, and contribution must have the same leading size; "
            f"got {count}, {usage.numel()}, {contribution.numel()}"
        )
    if atom_ids is None:
        atom_ids = torch.arange(count, dtype=torch.long)
    else:
        atom_ids = atom_ids.detach().cpu().to(torch.long).reshape(-1)
    if atom_ids.numel() != count:
        raise ValueError(
            f"atom_ids must contain {count} entries, got {atom_ids.numel()}"
        )
    if count == 0:
        return None

    projected, pc1_var, pc2_var = _pca_project_snapshots([atoms])
    points = projected[0]
    x_lim, y_lim = _fixed_square_axis_limits(projected)
    usage_np = usage.numpy()
    contribution_np = contribution.numpy()
    active = usage_np > 0
    active_count = int(active.sum())
    total_selections = float(usage_np.sum())
    active_usage = np.log1p(usage_np[active]) if active_count else np.empty(0)
    active_contribution = contribution_np[active] if active_count else np.empty(0)
    if active_count:
        size_scale = active_contribution / (active_contribution.max() + 1.0e-8)
        active_sizes = 20.0 + 90.0 * np.sqrt(size_scale)
    else:
        active_sizes = np.empty(0)

    fig, (ax_geom, ax_load) = plt.subplots(1, 2, figsize=(14, 6))
    ax_geom.scatter(
        points[:, 0],
        points[:, 1],
        c="#c7c7c7",
        s=8,
        alpha=0.28,
        linewidths=0,
        label="not selected in visual batch",
    )
    scatter = None
    if active_count:
        scatter = ax_geom.scatter(
            points[active, 0],
            points[active, 1],
            c=active_usage,
            s=active_sizes,
            cmap="viridis",
            alpha=0.9,
            linewidths=0.25,
            edgecolors="black",
            label="selected atom",
        )
        top_local = np.argsort(active_contribution)[-min(8, active_count):]
        active_indices = np.flatnonzero(active)
        for local_idx in top_local:
            point_idx = int(active_indices[local_idx])
            ax_geom.annotate(
                str(int(atom_ids[point_idx].item())),
                (points[point_idx, 0], points[point_idx, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#202020",
            )
    ax_geom.set_xlim(x_lim)
    ax_geom.set_ylim(y_lim)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax_geom.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax_geom.set_title("Atom geometry and validation usage")
    ax_geom.legend(loc="best", fontsize=8, frameon=False)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax_geom, fraction=0.045, pad=0.02)
        colorbar.set_label("log(1 + selection count)")

    if active_count:
        mean_magnitude = active_contribution / np.maximum(usage_np[active], 1.0)
        load_scatter = ax_load.scatter(
            usage_np[active],
            mean_magnitude,
            c=active_usage,
            s=active_sizes,
            cmap="viridis",
            alpha=0.82,
            linewidths=0.25,
            edgecolors="black",
        )
        ax_load.set_xscale("log")
        top_local = np.argsort(active_contribution)[-min(10, active_count):]
        active_ids = atom_ids[torch.from_numpy(np.flatnonzero(active)).long()].numpy()
        for local_idx in top_local:
            ax_load.annotate(
                str(int(active_ids[local_idx])),
                (usage_np[active][local_idx], mean_magnitude[local_idx]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        colorbar = fig.colorbar(load_scatter, ax=ax_load, fraction=0.045, pad=0.02)
        colorbar.set_label("log(1 + selection count)")
    else:
        ax_load.text(
            0.5,
            0.5,
            "No validation selections available",
            ha="center",
            va="center",
            transform=ax_load.transAxes,
        )
    ax_load.set_xlabel("Selection count (log scale)")
    ax_load.set_ylabel("Mean |coefficient|")
    ax_load.set_title("Active-atom load and contribution")
    ax_load.grid(True, alpha=0.2)

    fig.suptitle(
        f"{title} | step {int(step)} | active {active_count}/{count} sampled atoms | "
        f"{int(total_selections)} selections",
        fontsize=11,
    )
    fig.tight_layout()
    image = _figure_to_rgb_array(fig)
    plt.close(fig)
    return image


def render_dictionary_diagnostics(
    atoms: torch.Tensor,
    usage: torch.Tensor,
    contribution: torch.Tensor,
    *,
    atom_ids: torch.Tensor | None = None,
    step: int = 0,
    movement_snapshots: tuple[torch.Tensor, torch.Tensor] | None = None,
    movement_steps: tuple[int, int] | None = None,
    title: str = "Dictionary diagnostics",
):
    """Render geometry and load diagnostics without a lossy 2-D embedding.

    Pairwise and nearest-neighbor cosine statistics are measured in the actual
    atom space. Usage concentration is shown as cumulative mass, and aligned
    snapshots use angular drift rather than projected PCA displacement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    atoms = torch.nan_to_num(atoms.detach().cpu().to(torch.float32))
    usage = torch.nan_to_num(usage.detach().cpu().to(torch.float32)).reshape(-1)
    contribution = torch.nan_to_num(
        contribution.detach().cpu().to(torch.float32)
    ).reshape(-1)
    if atoms.ndim != 2:
        raise ValueError(f"Expected atoms [N,D], got {tuple(atoms.shape)}")
    count = int(atoms.size(0))
    if count == 0:
        return None
    if usage.numel() != count or contribution.numel() != count:
        raise ValueError(
            "atoms, usage, and contribution must have the same leading size; "
            f"got {count}, {usage.numel()}, {contribution.numel()}"
        )
    if atom_ids is None:
        atom_ids = torch.arange(count, dtype=torch.long)
    else:
        atom_ids = atom_ids.detach().cpu().to(torch.long).reshape(-1)
    if atom_ids.numel() != count:
        raise ValueError(f"atom_ids must contain {count} entries, got {atom_ids.numel()}")

    # Bound the quadratic similarity calculation while keeping a deterministic,
    # evenly spaced sample. All load-concentration panels still use every atom.
    similarity_count = min(count, 1024)
    similarity_idx = torch.linspace(0, count - 1, steps=similarity_count).round().long()
    similarity_atoms = torch.nn.functional.normalize(
        atoms.index_select(0, similarity_idx), p=2, dim=1, eps=1.0e-8
    )
    similarity_usage = usage.index_select(0, similarity_idx)
    cosine = (similarity_atoms @ similarity_atoms.t()).clamp(-1.0, 1.0)
    upper = torch.triu_indices(similarity_count, similarity_count, offset=1)
    pairwise = cosine[upper[0], upper[1]] if similarity_count > 1 else torch.zeros(1)
    cosine.fill_diagonal_(-float("inf"))
    nearest = cosine.max(dim=1).values if similarity_count > 1 else torch.zeros(1)
    active = similarity_usage > 0

    def _cdf(values: torch.Tensor):
        values = values[torch.isfinite(values)].sort().values
        if values.numel() == 0:
            return np.empty(0), np.empty(0)
        y = torch.arange(1, values.numel() + 1, dtype=torch.float32) / values.numel()
        return values.numpy(), y.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax_pairwise, ax_nearest, ax_concentration = axes[0]
    ax_top, ax_drift, ax_drift_cdf = axes[1]

    ax_pairwise.hist(pairwise.numpy(), bins=60, density=True, color="#4c78a8", alpha=0.85)
    ax_pairwise.axvline(float(pairwise.mean()), color="black", ls="--", lw=1.1)
    ax_pairwise.set_xlabel("atom-pair cosine similarity")
    ax_pairwise.set_ylabel("density")
    ax_pairwise.set_title(
        f"Pairwise similarity in {int(atoms.size(1))}D — mean {float(pairwise.mean()):.3f}"
    )
    ax_pairwise.grid(True, alpha=0.18)

    for mask, label, color in (
        (active, "selected in validation batch", "#e45756"),
        (~active, "not selected", "#72b7b2"),
    ):
        x_values, y_values = _cdf(nearest[mask])
        if x_values.size:
            ax_nearest.plot(x_values, y_values, label=label, color=color, lw=1.7)
    ax_nearest.set_xlabel("nearest-other-atom cosine similarity")
    ax_nearest.set_ylabel("empirical CDF")
    ax_nearest.set_title("Nearest-neighbor redundancy (actual atom space)")
    ax_nearest.legend(fontsize=8, loc="lower right")
    ax_nearest.grid(True, alpha=0.18)

    atom_fraction = torch.arange(1, count + 1, dtype=torch.float32) / count
    for values, label, color in (
        (usage, "selection count", "#f58518"),
        (contribution, "sum |coefficient|", "#54a24b"),
    ):
        ranked = values.sort(descending=True).values
        total = ranked.sum()
        cumulative = ranked.cumsum(0) / total.clamp_min(1.0e-8)
        ax_concentration.plot(atom_fraction.numpy(), cumulative.numpy(), label=label, color=color, lw=1.8)
    ax_concentration.plot([0, 1], [0, 1], color="#999999", ls=":", lw=1.0, label="uniform")
    ax_concentration.set_xlim(0.0, 1.0)
    ax_concentration.set_ylim(0.0, 1.02)
    ax_concentration.set_xlabel("fraction of atoms, ranked high to low")
    ax_concentration.set_ylabel("fraction of total mass captured")
    ax_concentration.set_title("Dictionary load concentration")
    ax_concentration.legend(fontsize=8, loc="lower right")
    ax_concentration.grid(True, alpha=0.18)

    top_count = min(16, count)
    top_ids = contribution.argsort(descending=True)[:top_count]
    top_usage = usage.index_select(0, top_ids)
    top_contribution = contribution.index_select(0, top_ids)
    selection_share = top_usage / usage.sum().clamp_min(1.0e-8)
    contribution_share = top_contribution / contribution.sum().clamp_min(1.0e-8)
    positions = np.arange(top_count)
    width = 0.42
    ax_top.bar(
        positions - width / 2,
        selection_share.numpy(),
        width,
        label="selection share",
        color="#f58518",
    )
    ax_top.bar(
        positions + width / 2,
        contribution_share.numpy(),
        width,
        label="|coefficient| share",
        color="#54a24b",
    )
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(
        [str(int(atom_ids[index])) for index in top_ids], rotation=60, ha="right", fontsize=7
    )
    ax_top.set_xlabel("atom ID (ranked by contribution)")
    ax_top.set_ylabel("share of validation total")
    ax_top.set_title("Highest-contributing atoms")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, axis="y", alpha=0.18)

    drift_degrees = torch.empty(0)
    drift_norm = torch.empty(0)
    if movement_snapshots is not None:
        first, latest = movement_snapshots
        first = torch.nan_to_num(first.detach().cpu().to(torch.float32))
        latest = torch.nan_to_num(latest.detach().cpu().to(torch.float32))
        if first.shape == latest.shape and first.ndim == 2 and first.numel() > 0:
            first_unit = torch.nn.functional.normalize(first, p=2, dim=1, eps=1.0e-8)
            latest_unit = torch.nn.functional.normalize(latest, p=2, dim=1, eps=1.0e-8)
            aligned_cosine = (first_unit * latest_unit).sum(dim=1).clamp(-1.0, 1.0)
            drift_degrees = torch.rad2deg(torch.acos(aligned_cosine))
            drift_norm = (latest - first).norm(dim=1)

    if drift_degrees.numel():
        ax_drift.hist(drift_degrees.numpy(), bins=50, color="#b279a2", alpha=0.85)
        median_angle = float(torch.quantile(drift_degrees, 0.5))
        p90_angle = float(torch.quantile(drift_degrees, 0.9))
        ax_drift.axvline(median_angle, color="black", ls="--", lw=1.0, label="median")
        ax_drift.axvline(p90_angle, color="#e45756", ls=":", lw=1.2, label="p90")
        ax_drift.set_title(f"Angular drift — median {median_angle:.2f}°, p90 {p90_angle:.2f}°")
        ax_drift.legend(fontsize=8)
        drift_x, drift_y = _cdf(drift_norm)
        ax_drift_cdf.plot(drift_x, drift_y, color="#9d755d", lw=1.8)
        step_text = ""
        if movement_steps is not None:
            step_text = f" (steps {int(movement_steps[0])}→{int(movement_steps[1])})"
        ax_drift_cdf.set_title(f"Aligned atom L2-drift CDF{step_text}")
    else:
        ax_drift.text(0.5, 0.5, "A second snapshot is needed", ha="center", va="center", transform=ax_drift.transAxes)
        ax_drift.set_title("Angular drift")
        ax_drift_cdf.text(0.5, 0.5, "A second snapshot is needed", ha="center", va="center", transform=ax_drift_cdf.transAxes)
        ax_drift_cdf.set_title("Aligned atom L2-drift CDF")
    ax_drift.set_xlabel("angle from initial atom (degrees)")
    ax_drift.set_ylabel("atom count")
    ax_drift.grid(True, alpha=0.18)
    ax_drift_cdf.set_xlabel("L2 displacement from initial atom")
    ax_drift_cdf.set_ylabel("empirical CDF")
    ax_drift_cdf.grid(True, alpha=0.18)

    active_count = int((usage > 0).sum())
    fig.suptitle(
        f"{title} | step {int(step)} | active {active_count}/{count} displayed atoms | "
        f"similarity sample {similarity_count}",
        fontsize=12,
    )
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
