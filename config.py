"""config.py — Stochastic Portfolio Theory engine configuration."""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-spt-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWF",
    "XSD",
    "XBI",
    "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": COMBINED_TICKERS,
}

# ── SPT Strategy Parameters ───────────────────────────────────────────────────
# Diversity-weighted
DIVERSITY_P = 0.5  # exponent — 0.5 = square-root weighting

# Rolling estimation windows
COV_WINDOW = 63  # covariance/variance estimation window (days)
SORTINO_EVAL_WINDOW = 63  # window for dynamic ensemble Sortino scoring

# Ensemble blend — weights updated every N days based on trailing Sortino
ENSEMBLE_REBL_FREQ = 21  # reweight ensemble monthly

# Output portfolio
MAX_ASSETS = 6  # top 5 ETFs + CASH
MIN_WEIGHT = 0.005  # lower floor gives optimiser more room to differentiate
MAX_WEIGHT = 0.45

# ── Date ─────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
