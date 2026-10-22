"""Prévision probabiliste de bougies avec Kronos : N trajectoires tracées sur un graphique.

Usage:
    python forecast.py --kronos-dir ../Kronos --symbol BTCUSDT --interval 1h
"""
import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
CONTEXT = 512


def fetch_klines(symbol, interval, limit=CONTEXT):
    r = requests.get(KLINES_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount",
            "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = df[c].astype(float)
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def load_predictor(kronos_dir, model_name, device):
    sys.path.insert(0, kronos_dir)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402 (repo shiyu-coder/Kronos)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
    return KronosPredictor(model, tokenizer, device=device, max_context=CONTEXT)


def sample_paths(predictor, df, step, pred_len, n_paths, T, top_p):
    """Un appel par trajectoire avec sample_count=1 : predict() moyenne les échantillons
    quand sample_count > 1, ce qui donnerait une seule courbe lissée au lieu d'un éventail."""
    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df["timestamps"]
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + step, periods=pred_len, freq=step))
    paths = []
    for i in range(n_paths):
        pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                 pred_len=pred_len, T=T, top_p=top_p, sample_count=1, verbose=False)
        paths.append(pred["close"].to_numpy())
        print(f"trajectoire {i + 1}/{n_paths}", end="\r")
    return y_ts, np.array(paths)


def regime(paths, last_close):
    up = float(np.mean(paths[:, -1] > last_close))
    if up >= 0.75 or up <= 0.25:
        label = "TRENDING " + ("haussier" if up >= 0.75 else "baissier")
    else:
        label = "CHOPPY"
    return up, label


def plot(df, y_ts, paths, symbol, interval, up, label, out):
    hist = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist["timestamps"], open=hist["open"], high=hist["high"],
                                 low=hist["low"], close=hist["close"], name="historique"))
    for p in paths:
        fig.add_trace(go.Scatter(x=y_ts, y=p, mode="lines", line=dict(width=1, color="rgba(255,165,0,0.25)"),
                                 showlegend=False, hoverinfo="skip"))
    lo, med, hi = np.percentile(paths, [10, 50, 90], axis=0)
    fig.add_trace(go.Scatter(x=y_ts, y=med, mode="lines", line=dict(width=3, color="orange"), name="médiane"))
    fig.add_trace(go.Scatter(x=y_ts, y=hi, mode="lines", line=dict(dash="dot", color="orange"), name="P90"))
    fig.add_trace(go.Scatter(x=y_ts, y=lo, mode="lines", line=dict(dash="dot", color="orange"), name="P10"))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title=f"{symbol} {interval} — {len(paths)} futurs Kronos | {label} ({up:.0%} des trajectoires finissent au-dessus)")
    fig.write_html(out)
    print(f"\nGraphique écrit dans {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kronos-dir", required=True, help="chemin du clone de shiyu-coder/Kronos")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--paths", type=int, default=30, help="100 est possible mais lent sur CPU")
    ap.add_argument("--model", default="Kronos-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="forecast.html")
    a = ap.parse_args()

    df = fetch_klines(a.symbol, a.interval)
    step = df["timestamps"].iloc[-1] - df["timestamps"].iloc[-2]
    predictor = load_predictor(a.kronos_dir, a.model, a.device)
    y_ts, paths = sample_paths(predictor, df, step, a.pred_len, a.paths, a.temperature, a.top_p)
    up, label = regime(paths, df["close"].iloc[-1])
    plot(df, y_ts, paths, a.symbol, a.interval, up, label, a.out)


if __name__ == "__main__":
    main()
