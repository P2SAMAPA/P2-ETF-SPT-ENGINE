"""engine.py — SPT engine: daily rebalance, dynamic ensemble, top-6 output."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from spt_strategies import (
    blend_weights,
    compute_sortino,
    diversity_weights,
    excess_growth_rate,
    max_entropy_weights,
    volatility_harvest_weights,
)


def _select_top(
    weights: np.ndarray,
    assets: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Keep top MAX_ASSETS by weight, re-normalise."""
    idx = np.argsort(weights)[::-1][: config.MAX_ASSETS]
    idx = np.sort(idx)
    w = weights[idx]
    w = np.clip(w, config.MIN_WEIGHT, config.MAX_WEIGHT)
    w /= w.sum()
    return w, [assets[i] for i in idx]


def run_spt(
    log_returns: pd.DataFrame,
    cash_returns: pd.Series,
    universe_tickers: list[str],
    universe_name: str,
) -> dict:
    """Run the SPT engine for one universe.

    Returns dict with weights_df, portfolio_returns, rebalances, summary.
    """
    avail = [t for t in universe_tickers if t in log_returns.columns]
    all_assets = avail + ["CASH"]
    n = len(all_assets)

    rets = pd.concat([log_returns[avail], cash_returns.rename("CASH")], axis=1).dropna()

    dates = rets.index
    T = len(dates)

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({len(avail)} ETFs + CASH)\n"
        f"Period: {dates[0].date()} → {dates[-1].date()}  ({T} days)\n"
        f"{'='*60}"
    )

    # ── Output containers ─────────────────────────────────────────────────────
    weights_records: list[dict] = []
    port_returns: list[float] = []
    rebalances: list[dict] = []

    # Strategy return trackers (for rolling Sortino)
    strat_rets: dict[str, list[float]] = {"div": [], "ent": [], "vol": []}

    # Current state
    current_weights: dict[str, float] = {a: 1.0 / n for a in all_assets}
    last_ensemble_update = -config.ENSEMBLE_REBL_FREQ

    # Ensemble Sortinos (initialise equal)
    s_div = s_ent = s_vol = 1.0

    for i, date in enumerate(dates):
        if i < config.COV_WINDOW:
            port_ret = sum(
                current_weights.get(a, 0.0) * rets.loc[date, a] for a in all_assets
            )
            port_returns.append(float(port_ret))
            weights_records.append({"date": date, **current_weights})
            for k in strat_rets:
                strat_rets[k].append(float(port_ret))
            continue

        # ── Rolling estimates ─────────────────────────────────────────────────
        window = rets[avail].iloc[i - config.COV_WINDOW : i]
        cov = window.cov().values * 252  # annualised covariance
        variances = np.diag(cov)  # annualised per-asset variance

        # CASH has near-zero variance — append
        variances_full = np.append(variances, 1e-8)
        cov_full = np.zeros((n, n))
        cov_full[: len(avail), : len(avail)] = cov

        # Equal-cap proxy weights (ETF universe has no market cap)
        mkt_weights = np.ones(n) / n

        # ── Compute three strategy weights ────────────────────────────────────
        w_div = diversity_weights(mkt_weights, p=config.DIVERSITY_P)
        w_ent = max_entropy_weights(n)
        w_vol = volatility_harvest_weights(variances_full)

        # ── Update ensemble blend every ENSEMBLE_REBL_FREQ days ──────────────
        if i - last_ensemble_update >= config.ENSEMBLE_REBL_FREQ:
            eval_window = config.SORTINO_EVAL_WINDOW
            if len(strat_rets["div"]) >= eval_window:
                s_div = compute_sortino(pd.Series(strat_rets["div"][-eval_window:]))
                s_ent = compute_sortino(pd.Series(strat_rets["ent"][-eval_window:]))
                s_vol = compute_sortino(pd.Series(strat_rets["vol"][-eval_window:]))
            last_ensemble_update = i

        # ── Blend weights ─────────────────────────────────────────────────────
        w_blend = blend_weights(w_div, w_ent, w_vol, s_div, s_ent, s_vol)

        # ── Select top MAX_ASSETS ─────────────────────────────────────────────
        w_top, top_assets = _select_top(w_blend, all_assets)
        current_weights = dict(zip(top_assets, w_top))

        # ── Compute per-strategy daily returns (for next Sortino eval) ────────
        day_ret_div = float(
            sum(w_div[j] * rets.loc[date, a] for j, a in enumerate(all_assets))
        )
        day_ret_ent = float(
            sum(w_ent[j] * rets.loc[date, a] for j, a in enumerate(all_assets))
        )
        day_ret_vol = float(
            sum(w_vol[j] * rets.loc[date, a] for j, a in enumerate(all_assets))
        )
        strat_rets["div"].append(day_ret_div)
        strat_rets["ent"].append(day_ret_ent)
        strat_rets["vol"].append(day_ret_vol)

        # ── Portfolio return (blended top-6) ──────────────────────────────────
        port_ret = sum(
            current_weights.get(a, 0.0) * rets.loc[date, a] for a in all_assets
        )
        port_returns.append(float(port_ret))

        # ── Record weights ────────────────────────────────────────────────────
        weights_records.append(
            {"date": date, **{a: current_weights.get(a, 0.0) for a in all_assets}}
        )

        # ── Excess growth rate diagnostic ─────────────────────────────────────
        top_idx = [all_assets.index(a) for a in top_assets]
        cov_top = cov_full[np.ix_(top_idx, top_idx)]
        egr = excess_growth_rate(w_top, cov_top)

        # ── Log rebalance ─────────────────────────────────────────────────────
        total_scores = max(s_div + s_ent + s_vol, 1e-8)
        ensemble_alloc = {
            "diversity": round(max(s_div, 0.0) / total_scores, 4),
            "max_entropy": round(max(s_ent, 0.0) / total_scores, 4),
            "vol_harvest": round(max(s_vol, 0.0) / total_scores, 4),
        }

        rebalances.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "assets": top_assets,
                "weights": w_top.tolist(),
                "excess_growth_rate": round(egr, 6),
                "sortino_div": round(s_div, 4),
                "sortino_ent": round(s_ent, 4),
                "sortino_vol": round(s_vol, 4),
                "ensemble_alloc": ensemble_alloc,
            }
        )

        if i % 252 == 0:
            print(
                f"  {date.date()}  top={top_assets}  "
                f"EGR={egr*100:.3f}%  "
                f"blend=div:{ensemble_alloc['diversity']:.2f}/"
                f"ent:{ensemble_alloc['max_entropy']:.2f}/"
                f"vol:{ensemble_alloc['vol_harvest']:.2f}"
            )

    weights_df = pd.DataFrame(weights_records).set_index("date").fillna(0.0)
    port_series = pd.Series(port_returns, index=dates, name="portfolio")

    # ── Summary stats ─────────────────────────────────────────────────────────
    ann_ret = float(port_series.mean() * 252)
    ann_vol = float(port_series.std() * np.sqrt(252))
    downside = port_series[port_series < 0]
    ds_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 1e-8
    sortino = ann_ret / (ds_std + 1e-8)
    cum = float(np.expm1(port_series.sum()))
    mdd = float(
        (
            (np.exp(port_series.cumsum()) - np.exp(port_series.cumsum()).cummax())
            / np.exp(port_series.cumsum()).cummax()
        ).min()
    )
    avg_egr = float(
        np.mean([r["excess_growth_rate"] for r in rebalances[config.COV_WINDOW :]])
    )

    print(
        f"\n  Summary → AnnRet={ann_ret*100:.2f}%  Vol={ann_vol*100:.2f}%  "
        f"Sortino={sortino:.2f}  MDD={mdd*100:.1f}%  "
        f"Cumulative={cum*100:.1f}%  AvgEGR={avg_egr*100:.4f}%/day"
    )

    return {
        "weights_df": weights_df,
        "portfolio_returns": port_series,
        "rebalances": rebalances,
        "summary": {
            "universe": universe_name,
            "ann_return": round(ann_ret, 6),
            "ann_vol": round(ann_vol, 6),
            "sortino": round(sortino, 4),
            "max_drawdown": round(mdd, 6),
            "cumulative_return": round(cum, 6),
            "avg_excess_growth_rate": round(avg_egr, 8),
            "n_rebalances": len(rebalances),
        },
    }
