from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results")
SUMMARY_DIR = ROOT / "gse280315_visiumhd_crc_summary"
FIG_DIR = SUMMARY_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "P1": {
        "stats": ROOT / "gse280315_visiumhd_crc_p1_128um_cellcom_smoke" / "lr_pair_associated_edge_statistics.csv",
        "filtered": ROOT / "gse280315_visiumhd_crc_p1_128um_cellcom_smoke" / "lr_communication_filtered_1.0.csv",
        "h5ad": ROOT / "gse280315_visiumhd_crc_p1_128um_inputs" / "GSM8594567_P1CRC_128um.h5ad",
    },
    "P2": {
        "stats": ROOT / "gse280315_visiumhd_crc_p2_128um_cellcom_smoke" / "lr_pair_associated_edge_statistics.csv",
        "filtered": ROOT / "gse280315_visiumhd_crc_p2_128um_cellcom_smoke" / "lr_communication_filtered_1.0.csv",
        "h5ad": ROOT / "gse280315_visiumhd_crc_p2_128um_inputs" / "GSM8594568_P2CRC_128um.h5ad",
    },
    "P5": {
        "stats": ROOT / "gse280315_visiumhd_crc_p5_128um_cellcom_smoke" / "lr_pair_associated_edge_statistics.csv",
        "filtered": ROOT / "gse280315_visiumhd_crc_p5_128um_cellcom_smoke" / "lr_communication_filtered_1.0.csv",
        "h5ad": ROOT / "gse280315_visiumhd_crc_p5_128um_inputs" / "GSM8594569_P5CRC_128um.h5ad",
    },
}

DOT_PAIRED = [
    "COL9A2_CD44",
    "COL9A2_SDC4",
    "LAMC2_CD44",
    "GDF15_TGFBR2",
    "COL9A2_SDC1",
    "LAMC2_DAG1",
    "LAMC2_ITGA6_ITGB4",
    "LAMB3_CD44",
    "LAMB3_ITGA6_ITGB1",
    "THBS3_SDC1",
    "THBS3_SDC4",
]

SPATIAL_PAIRS = ["COL9A2_SDC4", "LAMC2_CD44", "COL9A2_CD44", "GDF15_TGFBR2"]


def load_stats() -> pd.DataFrame:
    frames = []
    for sample, paths in SAMPLES.items():
        df = pd.read_csv(paths["stats"])
        df["sample"] = sample
        for col in ["supporting_unique_edges", "associated_edge_attention_mean", "total_lr_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_dotplot(stats: pd.DataFrame) -> None:
    plot_df = stats[stats["lr_pair"].isin(DOT_PAIRED)].copy()
    plot_df["lr_pair"] = pd.Categorical(plot_df["lr_pair"], categories=DOT_PAIRED[::-1], ordered=True)
    plot_df["sample"] = pd.Categorical(plot_df["sample"], categories=list(SAMPLES), ordered=True)
    plot_df = plot_df.sort_values(["lr_pair", "sample"])

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    x = plot_df["sample"].cat.codes
    y = plot_df["lr_pair"].cat.codes
    support = plot_df["supporting_unique_edges"].to_numpy()
    sizes = 25 + 260 * np.sqrt(support / np.nanmax(support))
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=plot_df["associated_edge_attention_mean"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xticks(range(len(SAMPLES)))
    ax.set_xticklabels(list(SAMPLES))
    ax.set_yticks(range(len(DOT_PAIRED)))
    ax.set_yticklabels(DOT_PAIRED[::-1])
    ax.set_xlabel("CRC Visium HD sample")
    ax.set_ylabel("")
    ax.set_title("Recurrent high-resolution LR programs", pad=10)
    ax.grid(axis="both", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("Mean associated-edge attention")

    handles = []
    for val in [1000, 2500, 5000]:
        handles.append(ax.scatter([], [], s=25 + 260 * np.sqrt(val / np.nanmax(support)), facecolor="white", edgecolor="black", label=str(val)))
    ax.legend(
        handles=handles,
        title="Unique edges",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.28, right=0.86, top=0.88, bottom=0.23)
    fig.savefig(FIG_DIR / "crc_visiumhd_recurrent_lr_dotplot.pdf")
    fig.savefig(FIG_DIR / "crc_visiumhd_recurrent_lr_dotplot.png", dpi=300)
    plt.close(fig)


def pair_spot_scores(filtered_csv: Path, pair: str) -> pd.Series:
    chunks = []
    usecols = ["src_spot_barcode", "lr_pair", "attention_score"]
    for chunk in pd.read_csv(filtered_csv, usecols=usecols, chunksize=250_000):
        sub = chunk[chunk["lr_pair"] == pair]
        if not sub.empty:
            chunks.append(sub[["src_spot_barcode", "attention_score"]])
    if not chunks:
        return pd.Series(dtype=float)
    df = pd.concat(chunks, ignore_index=True)
    return df.groupby("src_spot_barcode")["attention_score"].mean()


def plot_spatial() -> None:
    fig, axes = plt.subplots(
        nrows=len(SPATIAL_PAIRS),
        ncols=len(SAMPLES),
        figsize=(9.4, 9.2),
        constrained_layout=True,
    )
    for col_idx, (sample, paths) in enumerate(SAMPLES.items()):
        adata = ad.read_h5ad(paths["h5ad"])
        coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"])
        for row_idx, pair in enumerate(SPATIAL_PAIRS):
            ax = axes[row_idx, col_idx]
            scores = pair_spot_scores(paths["filtered"], pair)
            aligned = coords.join(scores.rename("score"), how="left")
            bg = aligned[aligned["score"].isna()]
            fg = aligned[aligned["score"].notna()]
            ax.scatter(bg["x"], bg["y"], s=2, c="#dddddd", linewidth=0)
            if not fg.empty:
                vals = fg["score"].to_numpy()
                vmax = np.percentile(vals, 98) if len(vals) > 5 else vals.max()
                ax.scatter(fg["x"], fg["y"], s=5, c=vals, cmap="magma", vmin=1.0, vmax=vmax, linewidth=0)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(sample)
            if col_idx == 0:
                ax.set_ylabel(pair, rotation=0, ha="right", va="center", labelpad=48)
    fig.suptitle("Spatial localization of recurrent LR attention", y=1.02)
    fig.savefig(FIG_DIR / "crc_visiumhd_recurrent_lr_spatial.pdf")
    fig.savefig(FIG_DIR / "crc_visiumhd_recurrent_lr_spatial.png", dpi=300)
    plt.close(fig)


def main() -> None:
    stats = load_stats()
    plot_dotplot(stats)
    plot_spatial()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
