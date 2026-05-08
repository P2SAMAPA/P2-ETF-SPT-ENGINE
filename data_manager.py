"""data_manager.py — Data loading for SPT engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ETF_TICKERS = sorted(set(config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (log_returns DataFrame, cash_returns Series)."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    available = [t for t in ETF_TICKERS if t in df.columns]
    prices = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if "TBILL_3M" in df.columns:
        tbill = df["TBILL_3M"].reindex(log_returns.index).ffill().fillna(0.0)
        cash_log = np.log1p(tbill / 100.0) / 252.0
    else:
        cash_log = pd.Series(0.0, index=log_returns.index)
    cash_log.name = "CASH"

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs "
        f"| CASH mean daily={cash_log.mean()*100:.4f}%"
    )
    return log_returns, cash_log
