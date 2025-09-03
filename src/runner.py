import argparse, yaml, time
from datetime import datetime
import pandas as pd
from .io_utils import load_all_csv, save_report
from .features import add_features
from .signals import make_signals
from .backtest import backtest
from .pump import add_pump_features

def summarize(bt: pd.DataFrame, cfg: dict) -> dict:
    if bt.empty: return {"status":"empty","ts":datetime.utcnow().isoformat()+"Z"}
    avg30 = bt["ret_30m"].mean() if "ret_30m" in bt else None
    hit = (bt.filter(like="ret_").ge(cfg["market"]["pnl_thresh"]).any(axis=1)).mean()
    return {"status":"ok","trades":len(bt),"avg_ret_30m":avg30,"hit>=pnl":float(hit),
            "ts":datetime.utcnow().isoformat()+"Z","trades_df":bt}

def once(cfg):
    df=load_all_csv(cfg["paths"]["data_dir"])
    if df.empty: save_report(cfg["paths"]["reports_dir"], {"status":"no_data"}); return
    df=add_features(df,cfg)
    df=add_pump_features(df,cfg) if cfg.get("strategy")=="pump" else make_signals(df)
    bt=backtest(df,cfg["market"]["horizons"])
    save_report(cfg["paths"]["reports_dir"], summarize(bt,cfg))

def loop_mode(cfg, mins):
    while True: once(cfg); time.sleep(mins*60)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report_every", type=int, default=None)
    args=ap.parse_args()
    cfg=yaml.safe_load(open("configs/config.yaml","r",encoding="utf-8"))
    if args.once: once(cfg)
    else:
        interval=args.report_every or cfg["report"]["interval_min"]
        loop_mode(cfg, interval)

if __name__ == "__main__": main()
