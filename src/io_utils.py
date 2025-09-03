import os, json
import pandas as pd
from datetime import datetime

def load_all_csv(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True); return pd.DataFrame()
    frames=[]
    for fn in os.listdir(d):
        if not fn.lower().endswith(".csv"): continue
        fp=os.path.join(d,fn)
        try:
            df=pd.read_csv(fp)
            need=["ts","open","high","low","close","volume"]
            if not all(c in df.columns for c in need): continue
            df["symbol"]=os.path.splitext(fn)[0]
            frames.append(df[need+["symbol"]].copy())
        except: pass
    if not frames: return pd.DataFrame()
    df=pd.concat(frames, ignore_index=True)
    df["ts"]=pd.to_datetime(df["ts"], unit="ms", errors="coerce")
    df=df.dropna(subset=["ts"]).sort_values(["symbol","ts"]).reset_index(drop=True)
    return df

def save_report(rep_dir, summary):
    os.makedirs(rep_dir, exist_ok=True)
    ts=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html=os.path.join(rep_dir,f"summary_{ts}.html")
    csv=os.path.join(rep_dir,f"trades_{ts}.csv")
    snap=os.path.join(rep_dir,"last_snapshot.txt")
    status=os.path.join(rep_dir,"last_status.json")
    with open(html,"w",encoding="utf-8") as f:
        f.write(f"<h2>Scanner summary {ts}</h2>")
        for k,v in summary.items():
            if k!="trades_df": f.write(f"<p><b>{k}</b>: {v}</p>")
    if "trades_df" in summary:
        summary["trades_df"].to_csv(csv, index=False, encoding="utf-8")
    with open(snap,"w",encoding="utf-8") as f: f.write(str(summary.get("status","ok")))
    with open(status,"w",encoding="utf-8") as f:
        json.dump({k:v for k,v in summary.items() if k!="trades_df"}, f, ensure_ascii=False, indent=2)
