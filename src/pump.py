import pandas as pd, numpy as np
def _z(s,win): m=s.rolling(win).mean(); sd=s.rolling(win).std(ddof=0); return (s-m)/sd.replace(0,np.nan)

def add_pump_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p=cfg["pump"]; out=[]
    for sym,g in df.groupby("symbol"):
        g=g.reset_index(drop=True).copy()
        z_win=p["z_win"]; sq=p["squeeze_win"]; nh=p["breakout_n_high"]; vr=p["vol_ramp_win"]
        g["vol_z"]=_z(g["volume"], z_win)
        std=g["close"].rolling(sq).std(ddof=0)
        g["bb_width"]=(4*std)/g["close"]
        g["squeeze"]=g["bb_width"]<p["squeeze_bb"]
        rmax=g["close"].rolling(p["near_upper_lookback"]).max()
        g["near_upper"]=g["close"]>=rmax*(1-p["near_upper_eps"])
        recent=g["volume"].rolling(vr).mean()
        prev=g["volume"].shift(vr).rolling(vr).mean()
        g["vol_ramp"]=(recent/prev)>=p["vol_ramp_ratio"]
        prior_max=g["close"].rolling(nh).max().shift(1)
        g["n_high_breakout"]=g["close"]>prior_max
        g["ret_1m"]=g["close"].pct_change().fillna(0)
        g["pump_label"]=(g["ret_1m"]>=p["pump_ret_1m"])&(g["vol_z"]>=p["vol_z_breakout"])
        pre_bits=pd.concat([g["squeeze"], g["near_upper"], g["bias"], (g["vol_ramp"]|(g["vol_z"]>1.0))], axis=1).astype(int)
        g["pre_score"]=pre_bits.sum(axis=1)
        g["pre_pump"]=g["pre_score"]>=p["pre_score_min"]
        if cfg["pump"]["entry"]=="pre":
            g["signal"]=g["pre_pump"]
        else:
            g["signal"]=g["n_high_breakout"]&(g["vol_z"]>=p["vol_z_breakout"])
        g["symbol"]=sym; out.append(g)
    return pd.concat(out, ignore_index=True)
