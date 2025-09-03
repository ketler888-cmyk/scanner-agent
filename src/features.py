import pandas as pd

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-prev).abs(),
        (df["low"]-prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, pd.NA)
    return 100 - (100/(1+rs))

def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = cfg["features"]; out=[]
    for sym,g in df.groupby("symbol"):
        g=g.reset_index(drop=True).copy()
        f=r["fast_sma"]; s=r["slow_sma"]; vwin=r["vol_spike_window"]
        g["fast_sma"]=g["close"].rolling(f, min_periods=f).mean()
        g["slow_sma"]=g["close"].rolling(s, min_periods=s).mean()
        g["rsi"]=rsi(g["close"], r["rsi_period"])
        g["vol_med"]=g["volume"].rolling(vwin, min_periods=vwin).median()
        g["vol_spike"]=g["volume"]>g["vol_med"]*r["vol_mult"]
        g["impulse"]=g["close"].pct_change().fillna(0)
        g["atr"]=_atr(g,14)
        g["bias"]=g["fast_sma"]>g["slow_sma"]
        g["symbol"]=sym
        out.append(g)
    return pd.concat(out, ignore_index=True)
