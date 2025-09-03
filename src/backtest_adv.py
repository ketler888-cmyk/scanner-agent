import pandas as pd

def backtest_time(df: pd.DataFrame, horizons):
    res = []
    for sym, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        idx = g.index[g["signal"]].tolist()
        for i in idx:
            entry = float(g.at[i,"close"])
            row = {"symbol": sym, "ts": g.at[i,"ts"], "entry": entry}
            for H in horizons:
                j = min(i + H, len(g) - 1)
                row[f"ret_{H}m"] = g.at[j,"close"] / entry - 1.0
            res.append(row)
    return pd.DataFrame(res)

def metrics(bt: pd.DataFrame, pnl_thresh: float):
    if bt.empty: return {"trades":0,"hit":0.0,"avg30":0.0,"score":0.0}
    hit = (bt.filter(like="ret_").ge(pnl_thresh).any(axis=1)).mean()
    avg30 = bt["ret_30m"].mean() if "ret_30m" in bt else 0.0
    score = float(hit) * (len(bt) ** 0.5)
    return {"trades":len(bt),"hit":float(hit),"avg30":float(avg30),"score":score}
