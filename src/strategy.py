import pandas as pd, numpy as np

def zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def make_pump_predictors(df: pd.DataFrame, Z_WIN, SQ_WIN, SQ_BB, BREAK_N, VOL_RAMP):
    g = df.copy()
    std = g["close"].rolling(SQ_WIN).std(ddof=0)
    g["bb_width"] = (4 * std) / g["close"]
    g["squeeze"] = g["bb_width"] < SQ_BB
    rmax = g["close"].rolling(60).max()
    g["near_upper"] = g["close"] >= rmax * (1 - 0.003)
    g["vol_z"] = zscore(g["volume"], Z_WIN)
    recent = g["volume"].rolling(30).mean()
    prev = g["volume"].shift(30).rolling(30).mean()
    g["vol_ramp_ok"] = (recent / prev) >= VOL_RAMP
    prior_max = g["close"].rolling(BREAK_N).max().shift(1)
    g["n_high_breakout"] = g["close"] > prior_max
    return g

def apply_expr(df: pd.DataFrame, expr: str, env: dict) -> pd.Series:
    return df.eval(expr, local_dict=env).astype(bool)
