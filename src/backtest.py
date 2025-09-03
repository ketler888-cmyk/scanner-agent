import pandas as pd
def backtest(df: pd.DataFrame, horizons):
    res=[]
    for sym,g in df.groupby("symbol"):
        g=g.reset_index(drop=True)
        idx=g.index[g["signal"]].tolist()
        for i in idx:
            e=float(g.at[i,"close"]); ts0=g.at[i,"ts"]; row={"symbol":sym,"ts":ts0,"entry":e}
            for H in horizons:
                j=min(i+H,len(g)-1); pnl=g.at[j,"close"]/e-1.0
                row[f"ret_{H}m"]=pnl
            res.append(row)
    return pd.DataFrame(res)
