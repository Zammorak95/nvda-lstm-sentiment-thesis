#!/usr/bin/env python3
"""
Thesis report for walk-forward trading predictions.

Input:
- walk_forward_oos_predictions.csv (from your walk-forward script)
  columns expected:
    date, y_true, y_prob_up, y_pred, threshold
  optional:
    target_next_return  (needed for trading/equity curve/stat tests)

Optional:
- walk_forward_summary.json (fold metrics)

Outputs:
- figures/*.png
- tables/*.csv
- report_summary.json

Includes:
- Equity curve + drawdown
- Rolling Sharpe
- Return distribution
- ROC curve + AUC
- Calibration plot + Brier score
- Trade-rate over time
- Fold metric plots (if summary.json exists)
- Statistical tests:
  - Mean return t-test
  - Newey-West HAC t-test (robust to autocorrelation)
  - Bootstrap Sharpe CI
  - Permutation test for strategy mean return (label shuffling)

Usage:
  source /home/zammorak/thesis/.venv/bin/activate
  python /home/zammorak/thesis/src/thesis/eval/thesis_walkforward_report.py \
    --oos models/walk_forward_direction_bestparams/walk_forward_oos_predictions.csv\
    --summary /home/zammorak/thesis/models/walk_forward_direction/walk_forward_summary.json \
    --outdir /home/zammorak/thesis/models/walk_forward_direction/thesis_report \
    --cost_bps 5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

# Optional statsmodels (recommended). Script still runs without it (just fewer tests).
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def annualized_sharpe(x: np.ndarray, periods: int = 252) -> float:
    x = np.asarray(x, dtype=float)
    std = x.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(x.mean() / std * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    def _sh(x):
        x = np.asarray(x, dtype=float)
        if x.std(ddof=0) == 0:
            return np.nan
        return x.mean() / x.std(ddof=0) * np.sqrt(252)
    return returns.rolling(window).apply(_sh, raw=False)


def add_costs(strategy_ret: np.ndarray, trade_flag: np.ndarray, cost_bps: float) -> np.ndarray:
    """
    Cost model (simple): cost applies on days you enter a position.
    For long-only next-day holding with y_pred as "enter long", this approximates 1 round-trip per trade.
    cost_bps: basis points per round-trip (e.g., 5 bps = 0.05%)
    """
    cost = (cost_bps / 10000.0)
    return strategy_ret - cost * trade_flag


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    n = len(r)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        boots[i] = annualized_sharpe(sample)
    return {
        "sharpe_boot_mean": float(np.nanmean(boots)),
        "sharpe_ci_2p5": float(np.nanpercentile(boots, 2.5)),
        "sharpe_ci_50": float(np.nanpercentile(boots, 50)),
        "sharpe_ci_97p5": float(np.nanpercentile(boots, 97.5)),
    }


def permutation_test_mean_return(strategy_ret: np.ndarray, trade_flag: np.ndarray, base_returns: np.ndarray,
                                 n_perm: int = 2000, seed: int = 123) -> dict:
    """
    Permutation test by shuffling trade decisions across time (breaks timing information).
    Keeps:
      - same number of trades
      - same base returns distribution
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(base_returns, dtype=float)
    f = np.asarray(trade_flag, dtype=int)
    obs = float(np.mean(strategy_ret))
    n = len(r)
    perms = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        f_perm = rng.permutation(f)
        perms[i] = float(np.mean(r * f_perm))
    p = float((np.sum(perms >= obs) + 1) / (n_perm + 1))
    return {"perm_obs_mean": obs, "perm_pvalue_one_sided": p}


def hac_ttest_mean_return(returns: np.ndarray, lags: int = 5) -> dict:
    """
    Newey-West HAC test for mean(returns) != 0.
    Requires statsmodels.
    """
    if not HAVE_SM:
        return {"hac_available": False}

    y = np.asarray(returns, dtype=float)
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(model.tvalues[0])
    p = float(model.pvalues[0])
    beta = float(model.params[0])
    se = float(model.bse[0])
    return {"hac_available": True, "hac_mean": beta, "hac_se": se, "hac_t": t, "hac_p": p, "hac_lags": lags}


def simple_ttest_mean_return(returns: np.ndarray) -> dict:
    """
    Classic t-test on mean return with iid assumption (often too optimistic for finance).
    """
    y = np.asarray(returns, dtype=float)
    n = len(y)
    mu = float(y.mean())
    sd = float(y.std(ddof=1))
    if sd == 0:
        return {"t_mean": mu, "t_stat": np.nan, "t_p_approx": np.nan, "n": n}

    t = mu / (sd / np.sqrt(n))

    # Normal approximation for p-value to avoid scipy dependency.
    # For large n this is fine.
    from math import erf, sqrt
    def norm_cdf(z):  # standard normal CDF
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    p_two = 2.0 * (1.0 - norm_cdf(abs(t)))

    return {"t_mean": mu, "t_stat": float(t), "t_p_approx": float(p_two), "n": n}


def plot_equity_drawdown(df: pd.DataFrame, out_png: str, title: str) -> dict:
    d = df.copy()
    d["equity"] = (1.0 + d["strategy_return"]).cumprod()
    d["peak"] = d["equity"].cummax()
    d["drawdown"] = d["equity"] / d["peak"] - 1.0

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(d["date"], d["equity"])
    ax1.set_title(f"{title} — Equity Curve")
    ax1.set_ylabel("Equity (normalized)")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(d["date"], d["drawdown"])
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    return {"max_drawdown": max_drawdown(d["equity"].values)}


def plot_rolling_sharpe(df: pd.DataFrame, out_png: str, window: int = 63, title: str = "") -> None:
    d = df.copy()
    rs = rolling_sharpe(d["strategy_return"], window=window)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["date"], rs)
    ax.set_title(f"{title} — Rolling Sharpe ({window}d)")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_return_hist(df: pd.DataFrame, out_png: str, title: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["strategy_return"].values, bins=60)
    ax.set_title(f"{title} — Strategy Daily Return Histogram")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_roc(df: pd.DataFrame, out_png: str, title: str) -> dict:
    y = df["y_true"].astype(int).values
    p = df["y_prob_up"].astype(float).values
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y, p)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"{title} — ROC (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    return {"auc": auc}


def plot_calibration(df: pd.DataFrame, out_png: str, n_bins: int = 10, title: str = "") -> dict:
    y = df["y_true"].astype(int).values
    p = df["y_prob_up"].astype(float).values
    brier = float(brier_score_loss(y, p))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    xs, ys, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        counts.append(int(m.sum()))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(xs, ys, marker="o")
    ax.set_title(f"{title} — Calibration (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of up")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    return {"brier": brier, "calib_bins_used": len(xs)}


def plot_trade_rate(df: pd.DataFrame, out_png: str, window: int = 63, title: str = "") -> None:
    d = df.copy()
    d["trade_flag"] = d["y_pred"].astype(int)
    tr = d["trade_flag"].rolling(window).mean()

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["date"], tr)
    ax.set_title(f"{title} — Rolling Trade Rate ({window}d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trade rate")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_fold_metrics(summary_json: str, out_png: str, metric: str = "test_auc") -> None:
    with open(summary_json, "r") as f:
        s = json.load(f)
    folds = s.get("folds", [])
    if not folds:
        return
    df = pd.DataFrame(folds)
    df["fold"] = df["fold"].astype(int)
    df = df.sort_values("fold")

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["fold"], df.get(metric, np.nan), marker="o")
    ax.set_title(f"Per-fold {metric}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True, help="Path to walk_forward_oos_predictions.csv")
    ap.add_argument("--summary", default=None, help="Path to walk_forward_summary.json (optional)")
    ap.add_argument("--outdir", required=True, help="Output directory for thesis figures/tables")
    ap.add_argument("--cost_bps", type=float, default=0.0, help="Round-trip cost in basis points per trade (e.g. 5 = 0.05%)")
    ap.add_argument("--rolling_window", type=int, default=63, help="Rolling window for Sharpe/trade-rate")
    ap.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap samples for Sharpe CI")
    ap.add_argument("--perm", type=int, default=2000, help="Permutation samples for mean-return test")
    ap.add_argument("--hac_lags", type=int, default=5, help="Newey-West lags for HAC test")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    figdir = os.path.join(args.outdir, "figures")
    tabdir = os.path.join(args.outdir, "tables")
    ensure_dir(figdir)
    ensure_dir(tabdir)

    df = pd.read_csv(args.oos)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = {"date", "y_true", "y_prob_up", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in OOS file: {sorted(missing)}")

    have_returns = "target_next_return" in df.columns
    if not have_returns:
        print("WARNING: target_next_return missing → no equity curve / trading stat tests will be produced.")

    # --- Classification metrics ---
    y = df["y_true"].astype(int).values
    p = df["y_prob_up"].astype(float).values
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    acc = float((df["y_pred"].astype(int).values == y).mean())
    trade_rate = float(df["y_pred"].astype(int).mean())

    # Plots: ROC & Calibration
    roc_info = plot_roc(df, os.path.join(figdir, "roc_curve.png"), title="Walk-forward OOS")
    calib_info = plot_calibration(df, os.path.join(figdir, "calibration.png"), title="Walk-forward OOS")
    plot_trade_rate(df, os.path.join(figdir, "rolling_trade_rate.png"), window=args.rolling_window, title="Walk-forward OOS")

    results = {
        "classification": {
            "oos_auc": auc,
            "oos_acc": acc,
            "trade_rate": trade_rate,
            "brier": calib_info.get("brier", None),
        },
        "cost_bps": args.cost_bps,
        "statsmodels_available": HAVE_SM,
    }

    # --- Trading metrics & stats if returns exist ---
    if have_returns:
        base_ret = df["target_next_return"].astype(float).values
        trade_flag = df["y_pred"].astype(int).values
        strat_ret = base_ret * trade_flag

        if args.cost_bps > 0:
            strat_ret_net = add_costs(strat_ret, trade_flag, args.cost_bps)
            df["strategy_return"] = strat_ret_net
            title = f"Strategy (net, cost={args.cost_bps:.1f} bps)"
        else:
            df["strategy_return"] = strat_ret
            title = "Strategy (gross, no costs)"

        # equity + drawdown
        dd_info = plot_equity_drawdown(df, os.path.join(figdir, "equity_drawdown.png"), title=title)
        plot_rolling_sharpe(df, os.path.join(figdir, "rolling_sharpe.png"), window=args.rolling_window, title=title)
        plot_return_hist(df, os.path.join(figdir, "return_hist.png"), title=title)

        sharpe = annualized_sharpe(df["strategy_return"].values)
        mdd = dd_info["max_drawdown"]

        # Stats tests
        ttest = simple_ttest_mean_return(df["strategy_return"].values)
        hac = hac_ttest_mean_return(df["strategy_return"].values, lags=args.hac_lags)
        boot = bootstrap_sharpe_ci(df["strategy_return"].values, n_boot=args.bootstrap)
        perm = permutation_test_mean_return(
            strategy_ret=df["strategy_return"].values,
            trade_flag=trade_flag,
            base_returns=base_ret,
            n_perm=args.perm
        )

        results["trading"] = {
            "annualized_sharpe": sharpe,
            "max_drawdown": mdd,
            "mean_daily_return": float(df["strategy_return"].mean()),
            "std_daily_return": float(df["strategy_return"].std(ddof=0)),
            "ttest": ttest,
            "hac_test": hac,
            "bootstrap_sharpe": boot,
            "permutation_test": perm,
            "num_days": int(len(df)),
            "num_trades": int(trade_flag.sum()),
        }

        # Save equity table for thesis
        out_table = df[["date", "y_true", "y_prob_up", "y_pred", "threshold", "target_next_return", "strategy_return"]].copy()
        out_table.to_csv(os.path.join(tabdir, "oos_predictions_with_strategy.csv"), index=False)

        # Yearly returns table
        tmp = df.copy()
        tmp["year"] = tmp["date"].dt.year
        yearly = tmp.groupby("year")["strategy_return"].apply(lambda x: float((1 + x).prod() - 1)).reset_index()
        yearly.to_csv(os.path.join(tabdir, "yearly_strategy_returns.csv"), index=False)

    # Fold plots if summary exists
    if args.summary and os.path.exists(args.summary):
        plot_fold_metrics(args.summary, os.path.join(figdir, "fold_test_auc.png"), metric="test_auc")
        plot_fold_metrics(args.summary, os.path.join(figdir, "fold_val_auc.png"), metric="val_auc")
        results["summary_json_used"] = args.summary

    # Save a single JSON report for your thesis appendix
    with open(os.path.join(args.outdir, "report_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== THESIS REPORT DONE ===")
    print("Output folder:", args.outdir)
    print("Key metrics:")
    print("  OOS AUC:", results["classification"]["oos_auc"])
    print("  OOS ACC:", results["classification"]["oos_acc"])
    print("  Trade rate:", results["classification"]["trade_rate"])
    if have_returns:
        print("  Sharpe:", results["trading"]["annualized_sharpe"])
        print("  Max Drawdown:", results["trading"]["max_drawdown"])


if __name__ == "__main__":
    main()