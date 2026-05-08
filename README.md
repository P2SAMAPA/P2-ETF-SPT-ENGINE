# 🎲 P2-ETF-STOCHASTIC-PORTFOLIO-THEORY

**P2Quant Engine** · Stochastic Portfolio Theory (SPT) with Dynamic Sortino Ensemble

[![Daily SPT Engine](https://github.com/P2SAMAPA/P2-ETF-STOCHASTIC-PORTFOLIO-THEORY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-STOCHASTIC-PORTFOLIO-THEORY/actions/workflows/daily_run.yml)

---

## What Is This?

This engine implements **Stochastic Portfolio Theory (SPT)**, developed by Robert Fernholz (1999–2002). SPT is a mathematical framework that generates provable excess returns over the market without requiring expected return forecasts — the alpha comes from the **mathematics of portfolio rebalancing itself**.

The core identity (Fernholz Master Equation):

```
g_p(t) = g_market(t) + γ_p(t)
```

Where `γ_p` is the **Excess Growth Rate** — a free lunch generated purely by diversification and rebalancing, proven to be non-negative under mild conditions.

---

## Three Sub-Strategies

### 🔵 Diversity-Weighted Portfolio
```
w_i ∝ μ_i^p,   p = 0.5
```
Overweights smaller/lower-weight assets relative to the market-cap benchmark. Generates a **diversity premium** — proven by Fernholz to outperform the cap-weighted index over time.

### 🟢 Maximum Entropy Portfolio
```
maximise H = -Σ wᵢ log(wᵢ)
```
Pushes weights toward maximum diversification. Mathematically guaranteed to generate positive excess growth rate by the entropy inequality.

### 🔴 Volatility Harvest Portfolio
```
maximise γ_p = ½ Σ wᵢ σᵢ²(1 - wᵢ)
```
Directly maximises the SPT excess growth rate by overweighting high-variance assets. Harvests the **variance drag** that causes individual assets to underperform their geometric mean.

---

## Dynamic Ensemble

The three strategies are blended daily using **trailing 63-day Sortino ratios** as weights:

```
blend_weight(strategy) = max(Sortino, 0) / Σ max(Sortino_i, 0)
```

- Strategies with negative Sortino receive **zero allocation**
- Falls back to equal-weight blend if all strategies are underperforming
- Ensemble reweighted every 21 trading days (~monthly)

---

## Portfolio Construction

| Parameter | Value |
|---|---|
| Max assets | 6 (5 ETFs + CASH) |
| Min weight | 1% |
| Max weight | 45% |
| Rebalance | Daily |
| Cov window | 63 days |
| Ensemble reweight | Every 21 days |
| CASH | 3M T-Bill (daily log return) |

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM + CASH |
| COMBINED | All above + TLT VCIT LQD HYG VNQ GLD SLV + CASH |

---

## Output

Results are pushed daily to HuggingFace:
- `spt_YYYY-MM-DD_{universe}.json` — summary, latest weights, rebalance log
- `portfolio_returns_{universe}.csv` — full daily return series 2008→today
- `weights_{universe}.csv` — full daily weight history

**Results repo:** [P2SAMAPA/p2-etf-spt-results](https://huggingface.co/datasets/P2SAMAPA/p2-etf-spt-results)

---

## Streamlit Dashboard

Five tabs:
1. **Current Portfolio** — pie chart, weight bars, ensemble strategy blend
2. **Performance** — cumulative return, rolling Sortino, drawdown, monthly heatmap
3. **SPT Diagnostics** — excess growth rate over time, ensemble blend history, per-strategy Sortino
4. **Weight History** — stacked area allocation chart, monthly weight heatmap
5. **Rebalance Log** — daily log with EGR, ensemble allocations, strategy Sortinos

---

## Data Source

| Field | Value |
|---|---|
| Input data | [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data) |
| Coverage | 2008-01-01 → present |
| ETF prices | Closing prices, forward-filled |
| CASH proxy | TBILL_3M (annualised %) → daily log return |

---

## References

- Fernholz, R. (1999). *On the diversity of equity markets*. Journal of Mathematical Economics.
- Fernholz, R. (2002). *Stochastic Portfolio Theory*. Springer.
- Fernholz, R. & Karatzas, I. (2009). *Stochastic Portfolio Theory: An Overview*. Handbook of Numerical Analysis.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
