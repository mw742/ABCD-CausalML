#!/usr/bin/env python3
import os, json, argparse, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def cluster_robust_se(beta, X, resid, clusters):
    """
    Sandwich SE for simple regression with clustering.
    X: (n, p) design used in final stage; include intercept if used.
    """
    # (X'X)^-1
    XtX_inv = np.linalg.pinv(X.T @ X)
    uniq = np.unique(clusters)
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for g in uniq:
        idx = np.where(clusters == g)[0]
        Xg = X[idx, :]
        ug = resid[idx].reshape(-1, 1)
        Sg = Xg.T @ ug
        meat += Sg @ Sg.T
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return se, V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/causal_dynamic/causal_panel_full.tsv")
    ap.add_argument("--edges", default="results/causal_dynamic/graph_granger/edges_stability.tsv")
    ap.add_argument("--outdir", default="results/causal_dynamic/effects_dml")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stability_min", type=float, default=0.6)
    ap.add_argument("--max_edges_per_target", type=int, default=30)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.panel, sep="\t")
    df["IID"] = df["IID"].astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    edges = pd.read_csv(args.edges, sep="\t")
    if edges.empty:
        raise ValueError("edges file is empty; run discovery first")

    # Keep stable edges only
    edges = edges[edges["stability"] >= args.stability_min].copy()
    if edges.empty:
        raise ValueError(f"No edges with stability >= {args.stability_min}")

    # Limit edges per target (prevents huge runtime)
    edges = edges.sort_values(["target", "stability"], ascending=[True, False])
    edges = edges.groupby("target", as_index=False).head(args.max_edges_per_target)

    y_cols = sorted([c for c in df.columns if c.startswith("y_")])
    results = []

    gkf = GroupKFold(n_splits=args.n_folds)

    # Use train+val to estimate effects; reserve test for prediction reporting separately
    dv = df[df["split"].isin(["train", "val"])].copy()

    for tgt, sub in edges.groupby("target"):
        if tgt not in y_cols:
            continue
        mcol = tgt.replace("y_", "m_", 1)
        rcol = tgt.replace("y_", "r_", 1)
        if mcol not in dv.columns or rcol not in dv.columns:
            continue

        ok_tgt = (dv[mcol].values.astype(float) == 1.0) & (dv["step"].values.astype(float) >= 1.0)
        if ok_tgt.sum() < 500:
            continue

        Y = dv.loc[ok_tgt, tgt].values.astype(float)
        groups = dv.loc[ok_tgt, "IID"].values

        # Precompute a covariate pool: all lagged variables up to L=max lag in predictor name
        # (This is conservative backdoor adjustment; you can later refine.)
        for _, row in sub.iterrows():
            pred = row["predictor"]
            stab = float(row["stability"])
            # pred should look like "var_L1" etc
            if "_L" not in pred:
                continue
            try:
                Lk = int(pred.split("_L")[-1])
            except Exception:
                Lk = 1

            # Treatment T = predictor value (continuous)
            if pred not in dv.columns:
                continue
            T = pd.to_numeric(dv.loc[ok_tgt, pred], errors="coerce").fillna(0.0).values.astype(float)

            # Covariates X: ALL lagged columns with lag <= Lk (excluding T itself), plus baseline-ish columns if present
            lag_cols = []
            for c in dv.columns:
                if "_L" in c:
                    try:
                        lag = int(c.split("_L")[-1])
                        if 1 <= lag <= Lk:
                            lag_cols.append(c)
                    except Exception:
                        pass

            Xcols = sorted(set(lag_cols) - {pred})

            if len(Xcols) < 5:
                continue

            Xraw = dv.loc[ok_tgt, Xcols].replace([np.inf, -np.inf], np.nan)
            Xraw = Xraw.apply(pd.to_numeric, errors="coerce")
            Xraw = Xraw.fillna(Xraw.median(numeric_only=True)).fillna(0.0).values

            # Standardize X for nuisance models
            sc = StandardScaler()
            X = sc.fit_transform(Xraw)

            # Cross-fitted nuisance models:
            # g(x)=E[Y|X], m(x)=E[T|X]
            y_hat = np.zeros_like(Y, dtype=float)
            t_hat = np.zeros_like(T, dtype=float)

            for tr_idx, te_idx in gkf.split(X, Y, groups=groups):
                Xtr, Xte = X[tr_idx], X[te_idx]
                Ytr = Y[tr_idx]
                Ttr = T[tr_idx]

                # Outcome model: random forest regressor (works for binary too as prob-ish)
                g = RandomForestRegressor(
                    n_estimators=300,
                    random_state=args.seed,
                    min_samples_leaf=20,
                    n_jobs=1
                )
                g.fit(Xtr, Ytr)
                y_hat[te_idx] = g.predict(Xte)

                # Treatment model: random forest regressor
                m = RandomForestRegressor(
                    n_estimators=300,
                    random_state=args.seed + 7,
                    min_samples_leaf=20,
                    n_jobs=1
                )
                m.fit(Xtr, Ttr)
                t_hat[te_idx] = m.predict(Xte)

            Yres = Y - y_hat
            Tres = T - t_hat

            # Final stage: Yres ~ a * Tres (with intercept)
            Z = np.column_stack([np.ones_like(Tres), Tres])
            beta = np.linalg.lstsq(Z, Yres, rcond=None)[0]
            resid = Yres - Z @ beta

            se, V = cluster_robust_se(beta, Z, resid, clusters=groups)
            a = float(beta[1])
            se_a = float(se[1])
            ci_lo = a - 1.96 * se_a
            ci_hi = a + 1.96 * se_a

            results.append({
                "target": tgt,
                "treatment": pred,
                "stability": stab,
                "n": int(len(Y)),
                "effect_per_1unit": a,
                "se_cluster": se_a,
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
                "note": "DML (RF nuisance) risk-difference scale approx"
            })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(args.outdir, "effects_dml.tsv"), sep="\t", index=False)

    with open(os.path.join(args.outdir, "run_manifest.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("[done] wrote:", args.outdir)

if __name__ == "__main__":
    main()

