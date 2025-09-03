import json, pandas as pd
from .features import add_features
from .strategy import make_pump_predictors, apply_expr

def eval_last(df: pd.DataFrame, strat_json: str) -> bool:
    s = json.load(open(strat_json,"r",encoding="utf-8"))
    p = s["best_params"]
    df = add_features(df, {"features":{"fast_sma":9,"slow_sma":21,"rsi_period":14,"vol_spike_window":60,"vol_mult":2.0}})
    g = make_pump_predictors(df, **p)
    sig = apply_expr(g, s["expr"], p)
    return bool(sig.iloc[-1])
