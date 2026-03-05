import pandas as pd
import numpy as np

def build_ew_portfolio_returns(coins_returns: pd.DataFrame, in_index: pd.DataFrame) -> pd.Series:
    """
    Build equally-weighted portfolio returns from CCI30 constituents.
    At each date, average the returns of coins that are in the index.
    
    r_EW(t) = (1/N) × Σᵢ r_i(t)

    Returns a Series of portfolio returns.
    """
    # Align in_index with coins_returns columns
    mask = in_index.reindex(columns=coins_returns.columns, index=coins_returns.index).fillna(False)
    masked_returns = coins_returns.where(mask)
    ew_returns = masked_returns.mean(axis=1).dropna()
    ew_returns.name = 'ew_portfolio'
    return ew_returns