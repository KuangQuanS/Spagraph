import argparse
import numpy as np
import pandas as pd


def load_composition(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.astype(float)


def compute_single_pcc(true_df: pd.DataFrame, pred_df: pd.DataFrame,
                       celltype: str, renormalize_pred: bool) -> float:
    shared_spots = true_df.index.intersection(pred_df.index)
    if len(shared_spots) == 0:
        raise ValueError("No overlapping spots between prediction and ground truth.")
    true_df = true_df.loc[shared_spots]
    pred_df = pred_df.loc[shared_spots]

    if celltype not in true_df.columns:
        raise ValueError(f"Cell type '{celltype}' not found in ground truth columns.")
    if celltype not in pred_df.columns:
        raise ValueError(f"Cell type '{celltype}' not found in prediction columns.")

    true_row_sum = true_df.sum(axis=1)
    true_row_sum[true_row_sum == 0] = 1.0
    true_vec = true_df[celltype] / true_row_sum

    if renormalize_pred:
        pred_row_sum = pred_df.sum(axis=1)
        pred_row_sum[pred_row_sum == 0] = 1.0
        pred_vec = pred_df[celltype] / pred_row_sum
    else:
        pred_vec = pred_df[celltype]
        row_sum = pred_df.sum(axis=1)
        if not np.allclose(row_sum.values, 1.0, atol=1e-3):
            diff = np.max(np.abs(row_sum.values - 1.0))
            print(f"Warning: predicted composition rows do not sum to 1 (max diff {diff:.4f}).")

    mask = np.isfinite(true_vec.values) & np.isfinite(pred_vec.values)
    if mask.sum() == 0:
        return np.nan
    t = true_vec.values[mask]
    p = pred_vec.values[mask]

    eps = 1e-8
    t_std = t.std()
    p_std = p.std()
    if t_std < eps or p_std < eps:
        return np.nan
    return np.mean((t - t.mean()) * (p - p.mean())) / (t_std * p_std)


def main():
    parser = argparse.ArgumentParser(description="Compute PCC for a single cell type.")
    parser.add_argument("--composition_pred_csv", required=True,
                        help="Predicted composition CSV (spots x celltypes).")
    parser.add_argument("--composition_true_csv", required=True,
                        help="Ground truth composition CSV (spots x celltypes).")
    parser.add_argument("--celltype", required=True,
                        help="Target cell type name to evaluate.")
    parser.add_argument("--output_csv", default="single_celltype_pcc.csv",
                        help="Where to save the PCC result (CSV with one row).")
    parser.add_argument("--renormalize_pred", action="store_true",
                        help="If set, row-normalize prediction before computing PCC.")
    args = parser.parse_args()

    true_df = load_composition(args.composition_true_csv)
    pred_df = load_composition(args.composition_pred_csv)

    pcc = compute_single_pcc(true_df, pred_df, args.celltype, args.renormalize_pred)
    result_df = pd.DataFrame([{
        "celltype": args.celltype,
        "pcc": pcc,
        "spots_used": len(true_df.index.intersection(pred_df.index))
    }])
    result_df.to_csv(args.output_csv, index=False)

    print(f"Aligned spots: {len(true_df.index.intersection(pred_df.index))}")
    print(f"PCC for '{args.celltype}': {pcc:.4f}" if not np.isnan(pcc) else
          f"PCC for '{args.celltype}': nan (constant or invalid vectors)")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
