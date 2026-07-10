from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd


KEY_PAIRS = [
    "DLL4_NOTCH1",
    "DLL4_NOTCH3",
    "TNC_SDC1",
    "TNN_SDC1",
    "TNN_SDC4",
    "PDGFD_PDGFRB",
    "LGALS9_PTPRC",
    "CD99_CD99",
    "CXCL12_CXCR4",
    "SPP1_CD44",
    "CDH1_CDH1",
]


KEY_GENES = [
    "DLL4",
    "NOTCH1",
    "NOTCH3",
    "NOTCH4",
    "JAG1",
    "TNN",
    "TNC",
    "SDC1",
    "SDC4",
    "PDGFD",
    "PDGFRB",
    "COL4A5",
    "LAMB1",
    "LAMB2",
    "LAMC2",
    "ITGA1",
    "ITGA2",
    "ITGAV",
    "ITGB1",
    "CDH1",
    "LGALS9",
    "PTPRC",
    "CD99",
]


def split_complex(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).replace("+", "_").split("_") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CellChat coverage in a 10x filtered_feature_bc_matrix.h5")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--cellchat", default="cellchat_human.csv")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as handle:
        genes = {name.decode().upper() for name in handle["matrix/features/name"][:]}
        shape = tuple(int(v) for v in handle["matrix/shape"][:])

    cellchat = pd.read_csv(args.cellchat)
    covered: set[str] = set()
    for _, row in cellchat.iterrows():
        ligand = split_complex(row["ligand"])
        receptor = split_complex(row["receptor"])
        if ligand and receptor and all(g in genes for g in ligand) and all(g in genes for g in receptor):
            covered.add(str(row["interaction_name"]))

    print(f"matrix_shape={shape}")
    print(f"genes={len(genes)}")
    print(f"covered_cellchat_pairs={len(covered)}")
    print("key_present=" + ",".join(g for g in KEY_GENES if g in genes))
    for pair in KEY_PAIRS:
        print(f"{pair}={pair in covered}")
    print("example_pairs=" + ",".join(sorted(covered)[:50]))


if __name__ == "__main__":
    main()
