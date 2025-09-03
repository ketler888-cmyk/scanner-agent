# было
def make_pump_predictors(df: pd.DataFrame, Z_WIN, SQ_WIN, SQ_BB, BREAK_N, VOL_RAMP):

# нужно
def make_pump_predictors(df: pd.DataFrame, **p):
    Z_WIN = p["Z_WIN"]; SQ_WIN = p["SQ_WIN"]; SQ_BB = p["SQ_BB"]
    BREAK_N = p["BREAK_N"]; VOL_RAMP = p["VOL_RAMP"]
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
