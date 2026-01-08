#!/usr/bin/env python3
import os, json, argparse, random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def is_binary_series(x: pd.Series) -> bool:
    vals = pd.unique(x.dropna())
    if len(vals) == 0:
        return False
    vals = set(map(float, vals))
    return vals.issubset({0.0, 1.0})

def fit_sparse_logistic(X, y, C=0.2, l1_ratio=0.8, max_iter=2000):
    # saga supports elasticnet for logistic regression
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        n_jobs=1,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results/causal_dynamic/causal_panel_train.tsv")
    ap.add_argument("--outdir", default="results/causal_dynamic/graph_granger")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max_lag", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--min_prev", type=float, default=0.01)
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--l1_ratio", type=float, default=0.8)
    ap.add_argument("--topk_per_target", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train, sep="\t")
    if "IID" not in df.columns:
        raise ValueError("train file must contain IID")
    df["IID"] = df["IID"].astype(str)

    y_cols = sorted([c for c in df.columns if c.startswith("y_") and "_L" not in c])
    m_cols = [c.replace("y_", "m_", 1) for c in y_cols]
    r_cols = [c.replace("y_", "r_", 1) for c in y_cols]

    for c in m_cols + r_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Candidate predictors = lagged features and lagged outcomes only (prevents simultaneity)
    pred_cols = []
    for c in df.columns:
        if c.endswith(tuple([f"_L{L}" for L in range(1, args.max_lag + 1)])):
            pred_cols.append(c)

    # Remove any IDs/splits/time keys if they slipped in
    bad_prefix = ("IID", "split", "tstart", "tstop", "step")
    pred_cols = [c for c in pred_cols if not c.startswith(bad_prefix)]

    if len(pred_cols) == 0:
        raise ValueError("No lagged predictors found. Did you run 01_build_causal_panel_dataset.R with max_lag>=1?")

    # Preprocess predictors (median impute, standardize)
    Xraw = df[pred_cols].replace([np.inf, -np.inf], np.nan)
    Xraw = Xraw.apply(pd.to_numeric, errors="coerce")
    Xraw = Xraw.fillna(Xraw.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw.values)

    # Bootstrap by subject for stability selection
    iids = df["IID"].unique()
    rng = np.random.RandomState(args.seed)

    # Store edge selection counts: (target, predictor) -> count
    counts = {}

    # For each target, also store a “one-fit” coefficient magnitude ranking (for top-k reporting)
    coef_rank = {}

    for ycol in y_cols:
        mcol = ycol.replace("y_", "m_", 1)
        rcol = ycol.replace("y_", "r_", 1)

        # Only at-risk + observed rows
        ok = (df[mcol].values.astype(float) == 1.0) & (df["step"].values.astype(float) >= 1.0)
        y = df.loc[ok, ycol].values.astype(float)
        if y.size < 500:
            continue
        prev = y.mean()
        if prev < args.min_prev or prev > 1 - args.min_prev:
            continue
        if np.unique(y).size < 2:
            continue

        Xok = X[ok]

        # One fit (for ranking)
        clf0 = fit_sparse_logistic(Xok, y.astype(int), C=args.C, l1_ratio=args.l1_ratio)
        coefs0 = clf0.coef_.ravel()
        nz = np.where(np.abs(coefs0) > 1e-12)[0]
        if nz.size > 0:
            ord_ix = nz[np.argsort(-np.abs(coefs0[nz]))]
            coef_rank[ycol] = [(pred_cols[j], float(coefs0[j])) for j in ord_ix[:args.topk_per_target]]
        else:
            coef_rank[ycol] = []

        # Bootstrap stability
        for b in range(args.n_boot):
            bs_iids = rng.choice(iids, size=len(iids), replace=True)
            bs_mask = df["IID"].isin(bs_iids).values & ok
            yb = df.loc[bs_mask, ycol].values.astype(int)
            if yb.size < 300 or np.unique(yb).size < 2:
                continue
            Xb = X[bs_mask]

            try:
                clf = fit_sparse_logistic(Xb, yb, C=args.C, l1_ratio=args.l1_ratio)
            except Exception:
                continue

            coefs = clf.coef_.ravel()
            sel = np.where(np.abs(coefs) > 1e-12)[0]
            for j in sel:
                key = (ycol, pred_cols[j])
                counts[key] = counts.get(key, 0) + 1

    # Build stability table
    rows = []
    for (tgt, pred), ct in counts.items():
        rows.append({
            "target": tgt,
            "predictor": pred,
            "boot_selected": int(ct),
            "boot_total": int(args.n_boot),
            "stability": float(ct / args.n_boot),
        })
    stab = pd.DataFrame(rows)
    if not stab.empty:
        stab = stab.sort_values(["target", "stability"], ascending=[True, False])
    stab.to_csv(os.path.join(args.outdir, "edges_stability.tsv"), sep="\t", index=False)

    # Save “top-k coefficients” report
    with open(os.path.join(args.outdir, "topk_coefficients.json"), "w") as f:
        json.dump(coef_rank, f, indent=2)

    # Save manifest
    manifest = vars(args)
    manifest["n_rows_train"] = int(df.shape[0])
    manifest["n_iids_train"] = int(len(iids))
    manifest["n_pred_cols"] = int(len(pred_cols))
    manifest["y_cols"] = y_cols
    with open(os.path.join(args.outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[done] wrote:", args.outdir)

if __name__ == "__main__":
    main()

