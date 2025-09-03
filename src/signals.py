import pandas as pd
def make_signals(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy()
    d["cross"]=d["fast_sma"]>d["slow_sma"]
    d["rsi_kick"]=(d["rsi"].shift(1)<40)&(d["rsi"]>50)
    d["impulse_ok"]=d["impulse"].abs()>0.015
    d["vol_ok"]=d["vol_spike"].fillna(False)
    d["signal"]=d["cross"]&d["rsi_kick"]&d["vol_ok"]&d["impulse_ok"]
    return d
