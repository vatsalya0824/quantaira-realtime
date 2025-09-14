import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests

# ───────────────────────── Setup
st.set_page_config(page_title="Quantaira Dashboard", layout="wide")
CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-realtime.onrender.com/api")
COLORS = {
    "pulse":        "#6F52ED",  # purple
    "spo2":         "#FF6B8A",  # pink
    "systolic_bp":  "#2D9CDB",  # blue
    "diastolic_bp": "#9B51E0",  # violet
}

# ───────────────────────── Helpers
def fetch_data(hours: int = 72) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/measurements", params={"hours": hours}, timeout=15)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc","metric","value_1","value_2"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse","heart_rate","hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2","sp02","oxygen"}), "metric"] = "spo2"

    bp_mask = mlow.isin({"blood_pressure","bp"})
    if "value_2" in df.columns and bp_mask.any():
        bp = df[bp_mask].copy()
        sys = bp.assign(metric="systolic_bp", value=pd.to_numeric(bp["value_1"], errors="coerce"))
        dia = bp.assign(metric="diastolic_bp", value=pd.to_numeric(bp["value_2"], errors="coerce"))
        df = pd.concat([df[~bp_mask], sys, dia], ignore_index=True)
    else:
        df["value"] = pd.to_numeric(df.get("value_1"), errors="coerce")

    # use only columns we chart
    return df[["timestamp_utc","metric","value"]].dropna(subset=["timestamp_utc","metric"])

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def nearest_y(sub: pd.DataFrame, t: pd.Timestamp) -> float | None:
    if sub.empty:
        return None
    diffs = np.abs(sub["timestamp_utc"].view("int64") - t.value)
    i = int(diffs.argmin())
    try:
        return float(sub.iloc[i]["value"])
    except Exception:
        return None

def stats_block(sub: pd.DataFrame) -> str:
    mu = float(sub["value"].mean()) if not sub.empty else float("nan")
    sigma = float(sub["value"].std(ddof=0)) if not sub.empty else float("nan")
    return f"<div class='stats-footer'><span>Mean (µ): {mu:.2f}</span><span>Sigma (σ): {sigma:.2f}</span></div>"

def plot_with_pill(df: pd.DataFrame, metric: str, color: str, pill_events: list[pd.Timestamp]):
    sub = df[df["metric"] == metric].sort_values("timestamp_utc")
    if sub.empty:
        st.warning(f"No data for {metric}")
        return

    fig = go.Figure()
    # gradient underlay
    fig.add_trace(go.Scatter(
        x=sub["timestamp_utc"], y=sub["value"],
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor=hex_to_rgba(color, 0.15),
        hoverinfo="skip", showlegend=False
    ))
    # main line
    fig.add_trace(go.Scatter(
        x=sub["timestamp_utc"], y=sub["value"],
        mode="lines", line=dict(width=2, color=color),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
        showlegend=False
    ))
    # pill markers on the line
    xs, ys = [], []
    for e in pill_events:
        v = nearest_y(sub, e)
        if v is not None: xs.append(e); ys.append(v)
    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(width=2, color="white")),
            hovertemplate="Pill opened<br>Value: %{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=20,r=20,t=10,b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="x unified", height=400
    )
    fig.update_xaxes(rangeslider=dict(visible=True), showgrid=True, gridcolor="rgba(120,120,180,0.20)", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(stats_block(sub), unsafe_allow_html=True)

# ───────────────────────── UI
with st.sidebar:
    st.markdown("<h2 class='sb-title'>Controls</h2>", unsafe_allow_html=True)
    hours = st.selectbox("Time window (hours)", [8, 24, 72, 168, 720], index=1)
    normalize = st.checkbox("Normalize Combined overlay", True)

st.markdown("<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>", unsafe_allow_html=True)

df = fetch_data(hours)
if df.empty:
    st.warning("No data yet.")
    st.stop()

# Pill events (from same stream if such metric exists)
pill_events = df.loc[df["metric"].str.contains("pillbox", case=False, na=False), "timestamp_utc"].tolist()

tabs = st.tabs(["❤️ Heart Rate", "💧 Systolic BP", "💜 Diastolic BP", "🫁 SpO₂", "📊 Combined"])
with tabs[0]: plot_with_pill(df, "pulse",        COLORS["pulse"],        pill_events)
with tabs[1]: plot_with_pill(df, "systolic_bp",  COLORS["systolic_bp"],  pill_events)
with tabs[2]: plot_with_pill(df, "diastolic_bp", COLORS["diastolic_bp"], pill_events)
with tabs[3]: plot_with_pill(df, "spo2",         COLORS["spo2"],         pill_events)

# ───────────── Combined Overlay (normalized option + pill markers per series)
with tabs[4]:
    show_cols = ["pulse", "spo2", "systolic_bp", "diastolic_bp"]
    combo = df[df["metric"].isin(show_cols)].sort_values("timestamp_utc").copy()
    if combo.empty:
        st.warning("No data for combined view")
    else:
        if normalize:
            combo["value"] = combo.groupby("metric")["value"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
            )

        figc = px.line(
            combo, x="timestamp_utc", y="value", color="metric",
            category_orders={"metric": show_cols},
            color_discrete_map=COLORS
        )
        figc.update_traces(line=dict(width=2))
        figc.update_layout(
            margin=dict(l=20,r=20,t=10,b=20),
            paper_bgcolor="white", plot_bgcolor="white",
            hovermode="x unified", height=420
        )
        figc.update_xaxes(rangeslider=dict(visible=True),
                          showgrid=True, gridcolor="rgba(120,120,180,0.20)", griddash="dot")
        figc.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

        # pill markers for each metric, placed at nearest sample
        for m in show_cols:
            sub_m = combo[combo["metric"] == m][["timestamp_utc","value"]]
            if sub_m.empty: continue
            xs, ys = [], []
            for e in pill_events:
                diffs = np.abs(sub_m["timestamp_utc"].view("int64") - e.value)
                i = int(diffs.argmin())
                xs.append(sub_m.iloc[i]["timestamp_utc"])
                ys.append(float(sub_m.iloc[i]["value"]))
            if xs:
                figc.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=10, color=COLORS[m], symbol="circle",
                                line=dict(width=2, color="white")),
                    name=f"Pill ({m})", showlegend=False,
                    hovertemplate=f"Pill opened • {m}<br>Value: %{y}"
                                  "<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>"
                ))

        st.plotly_chart(figc, use_container_width=True)
        # simple combined stats (overall)
        st.markdown(stats_block(combo), unsafe_allow_html=True)
