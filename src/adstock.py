import pandas as pd
import numpy as np

def adstock(series: pd.Series, lam: float) -> pd.Series:
    """
    Exponential carry-over (adstock) of a non-negative series.
    lam in [0,1). Higher = more persistence.
    """
    out = np.zeros(len(series), dtype=float)
    vals = series.to_numpy(dtype=float)
    if len(vals) == 0:
        return series.copy()
    out[0] = vals[0]
    for t in range(1, len(vals)):
        out[t] = vals[t] + lam * out[t-1]
    return pd.Series(out, index=series.index)
