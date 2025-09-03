import argparse, yaml, itertools, random, os, json
import pandas as pd
from .io_utils import load_all_csv
from .features import add_features
from .strategy import make_pump_predictors, apply_expr
from .backtest_adv import backtest_time, metrics

def trial(df, params, expr, horizons, pnl_thresh):
    g = make_pump_predictors(df, **params)
    g["signal"] = apply_expr(g, expr, {**params})
    bt = backtest_time(g, horizons)
    m = metrics(bt, pnl_thresh)
    m.update(params)
    return m, bt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="./reports/research")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    df = load_all_csv("./data")
    if df.empty:
        raise SystemExit("data/ is empty")

    # базовые фичи (МА, RSI и т.п.)
    df = add_features(df, {
        "features":{"fast_sma":9,"slow_sma":21,"rsi_period":14,"vol_spike_window":60,"vol_mult":2.0}
    })

    expr = cfg["expr"]; horizons = cfg["horizons"]; pnl = cfg["pnl_thresh"]

    grid = list(itertools.product(
        cfg["search"]["VOL_Z"],
        cfg["search"]["BREAK_N"],
        cfg["search"]["Z_WIN"],
        cfg["search"]["SQ_WIN"],
        cfg["search"]["SQ_BB"],
        cfg["search"]["VOL_RAMP"],
    ))
    random.shuffle(grid)
    grid = grid[:cfg.get("max_trials", 200)]

    rows = []; best = None; best_bt = None
    for VOL_Z, BREAK_N, Z_WIN, SQ_WIN, SQ_BB, VOL_RAMP in grid:
        params = dict(VOL_Z=VOL_Z, BREAK_N=BREAK_N, Z_WIN=Z_WIN, SQ_WIN=SQ_WIN, SQ_BB=SQ_BB, VOL_RAMP=VOL_RAMP)
        m, bt = trial(df, params, expr, horizons, pnl)
        rows.append(m)
        if not best or m["score"] > best["score"]:
            best, best_bt = m, bt

    pd.DataFrame(rows).sort_values("score", ascending=False)\
        .to_csv(os.path.join(args.out,"leaderboard.csv"), index=False, encoding="utf-8")
    if best_bt is not None:
        best_bt.to_csv(os.path.join(args.out,"best_trades.csv"), index=False, encoding="utf-8")
    json.dump({
        "name": cfg["name"],
        "expr": expr,
        "horizons": horizons,
        "pnl_thresh": pnl,
        "best_params": {k: best[k] for k in ["VOL_Z","BREAK_N","Z_WIN","SQ_WIN","SQ_BB","VOL_RAMP"]} if best else {}
    }, open(os.path.join(args.out,"best_strategy.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
