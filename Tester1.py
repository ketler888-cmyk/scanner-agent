# Tester1.py — сканер/бэктест «предпамп» паттернов
# Требует: pandas, numpy
# Запуск:  python Tester1.py            # одноразовый прогон
#          python Tester1.py --loop     # фоново, отчёт каждые 20 минут

import os
import sys
import time
import json
import glob
import math
import argparse
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------- Конфиг (можешь править под себя) -----------
DATA_DIR     = "data"        # папка с историей (CSV)
REPORTS_DIR  = "reports"     # сюда падают отчёты
TZ           = "UTC"         # таймзона данных
PNL_THRESH   = 0.50          # «памп» считаем, если доходность >= 50%
WINRATE_MIN  = 0.80          # целевой winrate (80%)
HORIZONS_MIN = [1, 10, 30, 60]  # горизонты прогноза (мин)
RSI_LEN      = 14
FAST_MA      = 9
SLOW_MA      = 21
VOL_SPIKE    = 2.0           # во сколько раз объём выше медианы окна
ROLL_MIN     = 60            # размер окна ресемплинга в минутах, если надо
# -------------------------------------------------------

os.makedirs(REPORTS_DIR, exist_ok=True)

def _find_col(cols: List[str], variants: List[str]) -> str:
    low = {c.lower(): c for c in cols}
    for v in variants:
        if v in low:
            return low[v]
    # пробуем частичное совпадение
    for c in cols:
        lc = c.lower()
        if any(v in lc for v in variants):
            return c
    raise KeyError(f"Не найдено поле из вариантов: {variants}")

def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    # Попытка найти колонку времени
    tcol = _find_col(df.columns.tolist(), ["time","timestamp","date","datetime"])
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    if ts.isna().all():
        # попробуем epoch (сек/мс)
        raw = df[tcol].astype(str).str.replace(r"[^0-9]", "", regex=True)
        arr = pd.to_numeric(raw, errors="coerce")
        if arr.max() > 1e12:
            ts = pd.to_datetime(arr, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(arr, unit="s", utc=True, errors="coerce")
    df = df.copy()
    df["__ts__"] = ts.dt.tz_convert(TZ) if ts.dt.tz is not None else ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    df = df.dropna(subset=["__ts__"]).sort_values("__ts__")
    df = df.set_index("__ts__")
    # Если данные не минутные — ресемплим до 1m
    if df.index.inferred_type != "datetime64":
        df.index = pd.to_datetime(df.index, utc=True)
    # проверим шаг
    if df.index.to_series().diff().median() > pd.Timedelta("2min"):
        df = df.resample("1min").last().ffill()
    return df

def _norm_schema(df: pd.DataFrame) -> pd.DataFrame:
    pcol = _find_col(df.columns.tolist(), ["close","price","close_price","c"])
    vcol = _find_col(df.columns.tolist(), ["volume","vol","quote_volume","v"])
    out = pd.DataFrame(index=df.index)
    out["price"]  = pd.to_numeric(df[pcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["price"])
    return out

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ma_fast"] = x["price"].rolling(FAST_MA).mean()
    x["ma_slow"] = x["price"].rolling(SLOW_MA).mean()
    x["rsi"]     = rsi(x["price"], RSI_LEN)
    x["vol_med"] = x["volume"].rolling(SLOW_MA).median()
    x["vol_spk"] = (x["volume"] / (x["vol_med"].replace(0,np.nan))).fillna(0)

    # сигналы (1 — есть сигнал):
    x["sig_ma_cross"] = ((x["ma_fast"] > x["ma_slow"]) & (x["ma_fast"].shift(1) <= x["ma_slow"].shift(1))).astype(int)
    x["sig_rsi_high"] = (x["rsi"] >= 70).astype(int)
    x["sig_vol_spike"]= (x["vol_spk"] >= VOL_SPIKE).astype(int)

    # совмещённый «предпамп»: хотя бы 2 условия
    x["signal"] = ((x["sig_ma_cross"] + x["sig_rsi_high"] + x["sig_vol_spike"]) >= 2).astype(int)
    return x

def future_return(prices: pd.Series, minutes: int) -> pd.Series:
    return (prices.shift(-minutes) / prices - 1.0)

def backtest(x: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    res = []
    for h in horizons:
        fut = future_return(x["price"], h)
        # успешная сделка — если через h минут PnL >= PNL_THRESH
        ok = (fut >= PNL_THRESH).astype(int)
        mask = x["signal"] == 1
        total = int(mask.sum())
        wins  = int((ok[mask]==1).sum())
        winrate = wins / total if total else 0.0
        avg_pnl = float(fut[mask].mean()) if total else 0.0
        res.append({
            "horizon_min": h,
            "trades": total,
            "wins": wins,
            "winrate": round(winrate, 4),
            "avg_pnl": round(avg_pnl, 4),
        })
    return pd.DataFrame(res)

def scan_file(path: str) -> Dict:
    try:
        raw = pd.read_csv(path)
        raw = _ensure_time_index(raw)
        df  = _norm_schema(raw)
        x   = build_signals(df)

        stats = backtest(x, HORIZONS_MIN)
        ok_any = bool((stats["winrate"] >= WINRATE_MIN).any() and (stats["trades"] > 0).any())
        summary = {
            "file": os.path.basename(path),
            "rows": int(len(df)),
            "now": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ok_by_any_rule": ok_any,
        }
        return {"summary": summary, "stats": stats, "last_row_ts": str(df.index[-1]) if len(df) else None}
    except Exception as e:
        return {"error": f"{os.path.basename(path)}: {e}"}

def load_all_csv(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))

def save_report(name: str, payload: Dict):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(REPORTS_DIR, f"{name}_{ts}")
    # json
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # md кратко
    lines = [f"# Tester1 report ({ts})",
             "",
             f"**Files scanned:** {payload.get('files_scanned',0)}",
             f"**OK assets (any rule):** {payload.get('ok_assets',0)}",
             ""]
    for item in payload.get("details", []):
        if "error" in item:
            lines.append(f"- ❌ {item['error']}")
        else:
            summ = item["summary"]
            lines.append(f"- ✅ {summ['file']} | rows={summ['rows']} | ok={summ['ok_by_any_rule']}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # csv агрегированный
    rows = []
    for item in payload.get("details", []):
        if "stats" in item:
            s = item["stats"].copy()
            s["file"] = item["summary"]["file"]
            rows.append(s)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(base + ".csv", index=False)

def run_once() -> Dict:
    files = load_all_csv(DATA_DIR)
    details = []
    for p in files:
        details.append(scan_file(p))
    ok_assets = 0
    for d in details:
        if "summary" in d and d["summary"]["ok_by_any_rule"]:
            ok_assets += 1
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files_scanned": len(files),
        "ok_assets": ok_assets,
        "config": {
            "PNL_THRESH": PNL_THRESH,
            "WINRATE_MIN": WINRATE_MIN,
            "HORIZONS_MIN": HORIZONS_MIN,
            "FAST_MA": FAST_MA,
            "SLOW_MA": SLOW_MA,
            "RSI_LEN": RSI_LEN,
            "VOL_SPIKE": VOL_SPIKE,
        },
        "details": details,
    }
    name = "tester1_report"
    save_report(name, payload)
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Запускать каждые 20 минут")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print(f"Нет папки {DATA_DIR}. Создаю…"); os.makedirs(DATA_DIR, exist_ok=True)
        print("Положи CSV с колонками времени, цены, объёма и запусти снова.")
        sys.exit(1)

    if args.loop:
        print("Tester1: циклический режим (каждые 20 минут). Ctrl+C для выхода.")
        while True:
            payload = run_once()
            ok = payload.get("ok_assets", 0)
            print(f"[{dt.datetime.now()}] Скан завершён. OK assets: {ok}/{payload.get('files_scanned',0)}")
            time.sleep(20 * 60)
    else:
        payload = run_once()
        ok = payload.get("ok_assets", 0)
        print(f"Готово. OK assets: {ok}/{payload.get('files_scanned',0)}. Отчёты см. в ./reports")

if __name__ == "__main__":
    main()
