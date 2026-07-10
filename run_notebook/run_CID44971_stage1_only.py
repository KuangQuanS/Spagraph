from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import umap

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spagraph.models.deconv_model import train_vae
from spagraph.models.stage1 import coEncoder


def _rename_umap_outputs(output_dir: Path, prefix: str) -> None:
    for index in (1, 2):
        for suffix in ("pdf", "png"):
            src = output_dir / f"modality_alignment_umap_{index}.{suffix}"
            dst = output_dir / f"{prefix}_modality_alignment_umap_{index}.{suffix}"
            if src.exists():
                shutil.move(str(src), str(dst))


def _compute_umap_coordinates(vae, data: np.ndarray, device) -> np.ndarray:
    vae.eval()
    embeddings = []
    with torch.no_grad():
        batch_size = 1000
        for start in range(0, len(data), batch_size):
            batch = torch.as_tensor(data[start:start + batch_size], dtype=torch.float32, device=device)
            mu, _ = vae.encoder(batch)
            embeddings.append(mu.cpu().numpy())
    latent = np.vstack(embeddings)
    n_neighbors = min(30, max(2, latent.shape[0] - 1))
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    knn.fit(latent)
    knn_dists, knn_indices = knn.kneighbors(latent)
    knn_dists = knn_dists[:, 1:]
    knn_indices = knn_indices[:, 1:]

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.3,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
        precomputed_knn=(knn_indices, knn_dists, None),
    )
    return reducer.fit_transform(latent)


def _style_seurat_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP_1", fontsize=12)
    ax.set_ylabel("UMAP_2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_modality_alignment_umap_seurat(
    vae,
    train_X: np.ndarray,
    train_modality: np.ndarray,
    output_dir: Path,
    device,
    y_train: np.ndarray | None = None,
) -> None:
    umap_coords = _compute_umap_coordinates(vae=vae, data=train_X, device=device)
    sc_mask = train_modality == 0
    st_mask = train_modality == 1

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.2))
    ax1.scatter(
        umap_coords[st_mask, 0],
        umap_coords[st_mask, 1],
        s=7,
        c="#F28E2B",
        alpha=0.8,
        linewidths=0,
        label="ST",
    )
    ax1.scatter(
        umap_coords[sc_mask, 0],
        umap_coords[sc_mask, 1],
        s=7,
        c="#4E79A7",
        alpha=0.8,
        linewidths=0,
        label="SC",
    )
    _style_seurat_axes(ax1, "Modality")
    ax1.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=2.0)
    _save_figure(fig1, output_dir, "modality_alignment_umap_1")

    fig2, ax2 = plt.subplots(figsize=(7.2, 6.2))
    if y_train is not None:
        sc_coords = umap_coords[sc_mask]
        unique_clusters = np.unique(y_train)
        palette = plt.cm.get_cmap("tab20", max(len(unique_clusters), 20))

        ax2.scatter(
            umap_coords[st_mask, 0],
            umap_coords[st_mask, 1],
            s=6,
            c="#D3D3D3",
            alpha=0.45,
            linewidths=0,
        )

        for idx, cluster_id in enumerate(unique_clusters):
            cluster_mask = y_train == cluster_id
            cluster_coords = sc_coords[cluster_mask]
            color = palette(idx % palette.N)
            ax2.scatter(
                cluster_coords[:, 0],
                cluster_coords[:, 1],
                s=7,
                c=[color],
                alpha=0.85,
                linewidths=0,
            )
            center = np.median(cluster_coords, axis=0)
            ax2.text(
                center[0],
                center[1],
                str(cluster_id),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            )
        _style_seurat_axes(ax2, "SC clusters + ST")
    else:
        ax2.scatter(
            umap_coords[st_mask, 0],
            umap_coords[st_mask, 1],
            s=7,
            c="#F28E2B",
            alpha=0.8,
            linewidths=0,
            label="ST",
        )
        ax2.scatter(
            umap_coords[sc_mask, 0],
            umap_coords[sc_mask, 1],
            s=7,
            c="#4E79A7",
            alpha=0.8,
            linewidths=0,
            label="SC",
        )
        _style_seurat_axes(ax2, "SC + ST")
        ax2.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=2.0)
    _save_figure(fig2, output_dir, "modality_alignment_umap_2")


def main() -> None:
    dataset_dir = _REPO_ROOT / "spagraph_data" / "database" / "Wu" / "CID44971"
    sc_file = dataset_dir / "CID44971_SC.h5ad"
    st_file = dataset_dir / "CID44971_ST.h5ad"
    output_dir = _REPO_ROOT / "spagraph_data" / "evaluate" / "CID44971_stage1_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = coEncoder(
        sc_file=str(sc_file),
        st_file=str(st_file),
        output_dir=str(output_dir),
        device=None,
        save_to_disk=True,
        seed=42,
    )

    top_n_per_type = 100
    resolution = 4.0
    batch_size = 512
    n_epochs = 300
    lr = 5e-4
    beta = 0.1
    hidden_dims = [512, 256]
    latent_dim = 256
    loss_type = "mse"
    lambda_mmd = 0.03
    use_dual_decoder = True
    marker_selection_method = "variance"
    print_every = 10

    sc_adata, st_adata = encoder.load_data()
    (
        train_X,
        test_X,
        train_modality,
        test_modality,
        y_train,
        y_test,
        _sc_X_final,
        _sc_all_genes_raw,
        _sc_all_labels,
    ) = encoder.prepare_marker_gene_data(
        sc_adata=sc_adata,
        st_adata=st_adata,
        top_n_per_type=top_n_per_type,
        resolution=resolution,
        precomputed_marker_file=None,
        marker_selection_method=marker_selection_method,
    )

    encoder.build_vae(
        input_dim=len(encoder.genes),
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        loss_type=loss_type,
        use_dual_decoder=use_dual_decoder,
    )

    plot_modality_alignment_umap_seurat(
        vae=encoder.vae,
        train_X=train_X,
        train_modality=train_modality,
        y_train=y_train,
        device=encoder.device,
        output_dir=output_dir,
    )
    _rename_umap_outputs(output_dir, "epoch0")

    train_labels = train_modality.copy()
    train_labels[: len(y_train)] = y_train
    train_labels[len(y_train) :] = -1

    test_labels = test_modality.copy()
    test_labels[: len(y_test)] = y_test
    test_labels[len(y_test) :] = -1

    best_loss = train_vae(
        vae=encoder.vae,
        train_X=train_X,
        test_X=test_X,
        train_modality=train_modality,
        test_modality=test_modality,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        beta=beta,
        loss_type=loss_type,
        lambda_mmd=lambda_mmd,
        device=encoder.device,
        output_dir=str(output_dir),
        print_every=print_every,
        patience=20,
        min_delta=1,
        train_labels=train_labels,
        test_labels=test_labels,
    )

    plot_modality_alignment_umap_seurat(
        vae=encoder.vae,
        train_X=train_X,
        train_modality=train_modality,
        y_train=y_train,
        device=encoder.device,
        output_dir=output_dir,
    )
    _rename_umap_outputs(output_dir, "final")

    summary_path = output_dir / "stage1_only_run_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Stage 1 only run completed.",
                f"SC file: {sc_file}",
                f"ST file: {st_file}",
                f"Output dir: {output_dir}",
                f"Best loss: {best_loss:.6f}",
                f"Epoch 0 plot 1: {output_dir / 'epoch0_modality_alignment_umap_1.pdf'}",
                f"Epoch 0 plot 2: {output_dir / 'epoch0_modality_alignment_umap_2.pdf'}",
                f"Final plot 1: {output_dir / 'final_modality_alignment_umap_1.pdf'}",
                f"Final plot 2: {output_dir / 'final_modality_alignment_umap_2.pdf'}",
                f"Training curves: {output_dir / 'vae_training_curves.pdf'}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
