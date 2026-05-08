"""app.py — Stochastic Portfolio Theory Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="SPT Engine · P2Quant",
    layout="wide",
    page_icon="🎲",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

COLOURS = [
    "#1B4F8A",
    "#27AE60",
    "#E74C3C",
    "#F39C12",
    "#8E44AD",
    "#148F77",
    "#CA6F1E",
    "#2471A3",
    "#B7950B",
    "#717D7E",
]
STRAT_COLOURS = {
    "diversity": "#1B4F8A",
    "max_entropy": "#27AE60",
    "vol_harvest": "#E74C3C",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading SPT results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading returns…")
def load_returns(universe: str) -> pd.DataFrame | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(
            f"{BASE_RAW}/portfolio_returns_{slug}.csv", headers=HEADERS, timeout=60
        )
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), parse_dates=["date"])
        return df.set_index("date")
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading weights…")
def load_weights(universe: str) -> pd.DataFrame | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(f"{BASE_RAW}/weights_{slug}.csv", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        return pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
    except Exception:
        return None


def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def colour(asset: str, i: int) -> str:
    return COLOURS[i % len(COLOURS)]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎲 Stochastic Portfolio Theory Engine")
st.caption(
    "Fernholz SPT · Three strategies: Diversity-Weighted · Max-Entropy · Volatility Harvest · "
    "Dynamic Sortino ensemble blend · Daily rebalance · Top 6 assets"
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**Tickers:** {len(config.UNIVERSES[universe])} ETFs + CASH")
    st.markdown(f"**Cov window:** {config.COV_WINDOW} days")
    st.markdown(f"**Ensemble reweight:** every {config.ENSEMBLE_REBL_FREQ} days")
    st.markdown(f"**Max assets:** {config.MAX_ASSETS}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

data = load_json(universe)
port_rets = load_returns(universe)
weights_hist = load_weights(universe)

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

summary = data.get("summary", {})
latest_weights = {k: v for k, v in data.get("latest_weights", {}).items() if v > 0.001}
latest_date = data.get("latest_date", "?")
rebalances = data.get("rebalances", [])
assets = sorted(latest_weights, key=latest_weights.get, reverse=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns(3)
h1.metric("Run Date", data.get("run_date", "?"))
h2.metric("Latest Date", latest_date)
h3.metric("Universe", universe)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ann. Return", fmt_pct(summary.get("ann_return", 0)))
k2.metric("Ann. Volatility", fmt_pct(summary.get("ann_vol", 0)))
k3.metric("Sortino", f"{summary.get('sortino', 0):.2f}")
k4.metric("Max Drawdown", fmt_pct(summary.get("max_drawdown", 0)))
k5.metric("Avg Daily EGR", f"{summary.get('avg_excess_growth_rate', 0)*100:.4f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Current Portfolio",
        "📈 Performance",
        "🔬 SPT Diagnostics",
        "🗂️ Weight History",
        "📋 Rebalance Log",
    ]
)

# ── Tab 1: Current Portfolio ──────────────────────────────────────────────────
with tab1:
    st.subheader(f"Portfolio as of {latest_date}")
    pie_col, bar_col = st.columns(2)

    with pie_col:
        fig_pie = go.Figure(
            go.Pie(
                labels=assets,
                values=[latest_weights[a] for a in assets],
                marker=dict(colors=[colour(a, i) for i, a in enumerate(assets)]),
                texttemplate="%{label}<br>%{percent}",
                hole=0.35,
            )
        )
        fig_pie.update_layout(
            title="Current Allocation",
            height=400,
            margin=dict(t=50, b=20, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie")

    with bar_col:
        fig_bar = go.Figure(
            go.Bar(
                y=assets,
                x=[latest_weights[a] * 100 for a in assets],
                orientation="h",
                marker_color=[colour(a, i) for i, a in enumerate(assets)],
                text=[f"{latest_weights[a]*100:.1f}%" for a in assets],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            title="Weight by Asset",
            xaxis_title="Weight (%)",
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(t=50, b=20, l=60, r=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar")

    # Latest ensemble allocation
    if rebalances:
        last = rebalances[-1]
        ea = last.get("ensemble_alloc", {})
        st.markdown("**Latest Ensemble Strategy Blend**")
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "🔵 Diversity-Weighted",
            f"{ea.get('diversity', 0):.1%}",
            help="Fernholz diversity portfolio — overweights smaller assets",
        )
        e2.metric(
            "🟢 Max-Entropy",
            f"{ea.get('max_entropy', 0):.1%}",
            help="Maximises portfolio entropy — maximum diversification",
        )
        e3.metric(
            "🔴 Vol-Harvest",
            f"{ea.get('vol_harvest', 0):.1%}",
            help="Maximises excess growth rate γ_p — harvests variance drag",
        )
        st.caption(
            f"Sortino scores → Diversity: {last.get('sortino_div', 0):.2f} · "
            f"Entropy: {last.get('sortino_ent', 0):.2f} · "
            f"VolHarvest: {last.get('sortino_vol', 0):.2f}"
        )

# ── Tab 2: Performance ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Portfolio Performance")
    if port_rets is not None and not port_rets.empty:
        daily = port_rets["portfolio_return"]
        cum_ret = (np.exp(daily.cumsum()) - 1) * 100

        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name="SPT Portfolio",
                line=dict(color="#1B4F8A", width=2),
                fill="tozeroy",
                fillcolor="rgba(27,79,138,0.1)",
            )
        )
        fig_perf.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_perf.update_layout(
            title=f"Cumulative Return — {universe}",
            yaxis_title="Cumulative Return (%)",
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_perf, use_container_width=True, key="perf")

        c1, c2 = st.columns(2)
        with c1:
            # Rolling Sortino
            roll = 63
            rm = daily.rolling(roll).mean() * 252
            rd = daily.rolling(roll).apply(
                lambda x: x[x < 0].std() * np.sqrt(252) if (x < 0).any() else 1e-8
            )
            rs = rm / (rd + 1e-8)
            fig_rs = go.Figure(
                go.Scatter(
                    x=rs.index,
                    y=rs.values,
                    mode="lines",
                    line=dict(color="#27AE60", width=1.5),
                    name="Sortino",
                )
            )
            fig_rs.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_rs.add_hline(
                y=1, line_dash="dash", line_color="#F39C12", annotation_text="Sortino=1"
            )
            fig_rs.update_layout(
                title="Rolling 63-day Sortino",
                height=280,
                margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rs, use_container_width=True, key="sortino")

        with c2:
            # Drawdown
            cum = np.exp(daily.cumsum())
            dd = (cum - cum.cummax()) / cum.cummax() * 100
            fig_dd = go.Figure(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.3)",
                    line=dict(color="#E74C3C", width=1),
                    name="Drawdown",
                )
            )
            fig_dd.update_layout(
                title="Drawdown (%)",
                height=280,
                margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, use_container_width=True, key="dd")

        # Monthly heatmap
        monthly = daily.resample("ME").sum()
        mdf = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "ret": monthly.values * 100,
            }
        )
        pivot = mdf.pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(pivot.columns)]
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Return %"),
                text=[
                    [f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                    for row in pivot.values
                ],
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig_hm.update_layout(
            title="Monthly Returns Heatmap (%)",
            height=max(300, len(pivot) * 28 + 80),
            margin=dict(t=40, b=40, l=60, r=20),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="monthly_hm")
    else:
        st.info("No portfolio return data found.")

# ── Tab 3: SPT Diagnostics ────────────────────────────────────────────────────
with tab3:
    st.subheader("SPT Diagnostics — Excess Growth Rate & Ensemble Blend")
    st.caption(
        "Excess Growth Rate (EGR) = γ_p = ½(Σwᵢσᵢ² − σ_p²). "
        "This is the SPT 'free lunch' — generated purely by rebalancing, "
        "independent of expected returns."
    )

    if rebalances:
        reb_dates = [r["date"] for r in rebalances]
        egr_vals = [r["excess_growth_rate"] * 100 for r in rebalances]

        fig_egr = go.Figure(
            go.Scatter(
                x=reb_dates,
                y=egr_vals,
                mode="lines",
                line=dict(color="#8E44AD", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(142,68,173,0.15)",
                name="EGR (%/day)",
            )
        )
        fig_egr.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_egr.update_layout(
            title="Daily Excess Growth Rate γ_p (annualised basis points)",
            yaxis_title="EGR (%/day)",
            height=300,
            margin=dict(t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_egr, use_container_width=True, key="egr")

        # Ensemble blend over time
        div_alloc = [r["ensemble_alloc"].get("diversity", 0) for r in rebalances]
        ent_alloc = [r["ensemble_alloc"].get("max_entropy", 0) for r in rebalances]
        vol_alloc = [r["ensemble_alloc"].get("vol_harvest", 0) for r in rebalances]

        fig_blend = go.Figure()
        for name, vals, col in [
            ("Diversity", div_alloc, STRAT_COLOURS["diversity"]),
            ("Max-Entropy", ent_alloc, STRAT_COLOURS["max_entropy"]),
            ("Vol-Harvest", vol_alloc, STRAT_COLOURS["vol_harvest"]),
        ]:
            fig_blend.add_trace(
                go.Scatter(
                    x=reb_dates,
                    y=[v * 100 for v in vals],
                    mode="lines",
                    stackgroup="one",
                    name=name,
                    line=dict(width=0.5, color=col),
                    fillcolor=col,
                )
            )
        fig_blend.update_layout(
            title="Ensemble Strategy Blend Over Time (%)",
            yaxis_title="Allocation (%)",
            height=320,
            margin=dict(t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_blend, use_container_width=True, key="blend_chart")

        # Per-strategy Sortino over time
        fig_sort_strat = go.Figure()
        for name, key, col in [
            ("Diversity", "sortino_div", STRAT_COLOURS["diversity"]),
            ("Max-Entropy", "sortino_ent", STRAT_COLOURS["max_entropy"]),
            ("Vol-Harvest", "sortino_vol", STRAT_COLOURS["vol_harvest"]),
        ]:
            vals = [r.get(key, 0) for r in rebalances]
            fig_sort_strat.add_trace(
                go.Scatter(
                    x=reb_dates,
                    y=vals,
                    mode="lines",
                    name=name,
                    line=dict(color=col, width=1.5),
                )
            )
        fig_sort_strat.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_sort_strat.update_layout(
            title="Per-Strategy Trailing Sortino Ratio",
            yaxis_title="Sortino",
            height=300,
            margin=dict(t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sort_strat, use_container_width=True, key="strat_sortino")
    else:
        st.info("No rebalance data found.")

# ── Tab 4: Weight History ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Weight History")
    if weights_hist is not None and not weights_hist.empty:
        w = weights_hist.loc[:, (weights_hist > 0.001).any()]
        fig_area = go.Figure()
        for i, col in enumerate(w.columns):
            fig_area.add_trace(
                go.Scatter(
                    x=w.index,
                    y=w[col] * 100,
                    mode="lines",
                    stackgroup="one",
                    name=col,
                    line=dict(width=0.5, color=colour(col, i)),
                    fillcolor=colour(col, i),
                )
            )
        fig_area.update_layout(
            title=f"Portfolio Allocation Over Time — {universe}",
            yaxis_title="Weight (%)",
            xaxis_title="Date",
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_area, use_container_width=True, key="area")

        # Rebalance heatmap (sampled monthly)
        if rebalances:
            monthly_reb = rebalances[::21]  # ~monthly sample
            all_reb_assets = sorted(set(a for r in monthly_reb for a in r["assets"]))
            heat_z = []
            heat_dates = []
            for r in monthly_reb:
                w_map = dict(zip(r["assets"], r["weights"]))
                heat_z.append([w_map.get(a, 0.0) * 100 for a in all_reb_assets])
                heat_dates.append(r["date"])
            fig_reb_hm = go.Figure(
                go.Heatmap(
                    z=heat_z,
                    x=all_reb_assets,
                    y=heat_dates,
                    colorscale="Blues",
                    colorbar=dict(title="Weight %"),
                    text=[[f"{v:.0f}%" for v in row] for row in heat_z],
                    texttemplate="%{text}",
                    hoverongaps=False,
                )
            )
            fig_reb_hm.update_layout(
                title="Monthly Weight Snapshot (sampled)",
                height=max(300, len(heat_dates) * 16 + 80),
                margin=dict(t=40, b=60, l=80, r=20),
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_reb_hm, use_container_width=True, key="reb_hm")
    else:
        st.info("No weight history found.")

# ── Tab 5: Rebalance Log ──────────────────────────────────────────────────────
with tab5:
    st.subheader("Rebalance Log (last 252 days)")
    if rebalances:
        rows = []
        for r in reversed(rebalances[-100:]):
            ea = r.get("ensemble_alloc", {})
            rows.append(
                {
                    "Date": r["date"],
                    "Assets": ", ".join(r["assets"]),
                    "EGR": f"{r.get('excess_growth_rate', 0)*100:.4f}%",
                    "Div%": f"{ea.get('diversity', 0):.0%}",
                    "Ent%": f"{ea.get('max_entropy', 0):.0%}",
                    "Vol%": f"{ea.get('vol_harvest', 0):.0%}",
                    "S_div": f"{r.get('sortino_div', 0):.2f}",
                    "S_ent": f"{r.get('sortino_ent', 0):.2f}",
                    "S_vol": f"{r.get('sortino_vol', 0):.2f}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.info("No rebalance data found.")

st.divider()
st.caption(
    f"P2Quant SPT Engine · Run: {data.get('run_date', '?')} · "
    f"Fernholz (1999) · Data: {config.HF_DATA_REPO}"
)
