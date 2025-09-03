import os
import pandas as pd
import numpy as np
from datetime import datetime

# Папка с историей котировок
DATA_DIR = "data"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Параметры фильтрации
PUMP_THRESHOLD = 0.5   # 50% P&L
WINRATE_MIN = 0.8      # 80% успешных сделок

def load_data():
    all_data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(path)
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def analyze_data(df):
    if df.empty:
        return {"status": "Нет данных"}
    
    # Допустим, у нас есть столбцы: time, price, pnl
    df["signal"] = (df["pnl"] > PUMP_THRESHOLD) & (df["success_rate"] > WINRATE_MIN)
    signals = df[df["signal"] == True]

    result = {
        "total_records": len(df),
        "pump_signals": len(signals),
        "pump_percentage": len(signals) / len(df) * 100
    }
    return result

def save_report(stats):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"report_{ts}.txt")
    with open(report_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Отчёт сохранён: {report_path}")

if __name__ == "__main__":
    data = load_data()
    stats = analyze_data(data)
    save_report(stats)
    print("Готово! Результаты смотри в папке reports.")
