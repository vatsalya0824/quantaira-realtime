import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
TIME_WINDOW_HOURS = int(os.environ.get("TIME_WINDOW_HOURS", "72"))

st.set_page_config(page_title="Quantaira Dashboard", layout="wide")
st.title("Quantaira Dashboard")
st.caption(f"Source: {API_BASE} • Window: {TIME_WINDOW_HOURS}h")

@st.cache_data(ttl=10, show_spinner=False)
def fetch_data(hours: int):
    r = requests.get(f"{API_BASE}/api/measurements", params={"hours": hours}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    # normalize to columns used by charts
    df["metric"] = df["metric"].str.lower()
    df["value"] = df["value_1"]
    # normalize aliases
    df.loc[df["metric"].isin(["spO2","spo2","sp_o2","oxygen"]), "metric"] = "spo2"
    df.loc[df["metric"].isin(["pulse","heart_rate","hr"]), "metric"] = "pulse"
    # expand BP if needed (if you ever store as blood_pressure)
    # already normalized in backend as rows – but safe to keep:
    return df.sort_values("created_utc")

df = fetch_data(TIME_WINDOW_HOURS)
if df.empty:
    st.warning("No data in window yet. Send a Tenovi test payload or wait for live readings.")
    st.stop()

# timezone & UI
tz_choice = st.selectbox("Timezone", ["UTC", "America/New_York", "Europe/London", "Asia/Kolkata"], index=0)
df["local_time"] = df["created_utc"].dt.tz_convert(tz_choice)

# pill events = rows where metric == 'pillbox_opened'
pill_events = df[df["metric"] == "pillbox_opened"]["local_time"].tolist()

def add_series(fig, sub, color, name):
    # area fill
    fig.add_trace(
        go.Scatter(
            x=sub["local_time"], y=sub["value"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0,0,0,0.0)",  # no gradient in combined
            hoverinfo="skip",
            showlegend=False,
            name=""
        )
    )
    # main line
    fig.add_trace(
        go.Scatter(
            x=sub["local_time"], y=sub["value"],
            mode="lines",
            line=dict(width=2, color=color),
            name=name,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"+name+": %{y}<extra></extra>",
        )
    )

def add_pill_markers(fig, sub, color):
    # Find y near each pill opening and put markers
    if sub.empty or not pill_events:
        return
    times = sub["local_time"]
    vals = sub["value"]
    for e in pill_events:
        # nearest index by absolute time difference
        idx = (times - e).abs().argmin()
        fig.add_trace(
            go.Scatter(
                x=[times.iloc[idx]], y=[vals.iloc[idx]],
                mode="markers",
                marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color="white")),
                name="Pill opened",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Pill opened near reading: %{y}<extra></extra>",
                showlegend=False,
            )
        )

def fig_layout(fig):
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(120,120,180,0.20)", griddash="dot")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20), paper_bgcolor="white", plot_bgcolor="white")

tabs = st.tabs(["❤️ Heart Rate", "💧 Systolic BP", "💜 Diastolic BP", "🫁 SpO₂", "📊 Combined"])

COLORS = {
    "pulse":        "#6F52ED",
    "spo2":         "#FF6B8A",
    "systolic_bp":  "#2D9CDB",
    "diastolic_bp": "#9B51E0",
}

with tabs[0]:
    sub = df[df["metric"]=="pulse"]
    st.subheader("Heart Rate")
    fig = go.Figure()
    add_series(fig, sub, COLORS["pulse"], "Heart Rate (bpm)")
    add_pill_markers(fig, sub, COLORS["pulse"])
    fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    sub = df[df["metric"]=="systolic_bp"]
    st.subheader("Systolic BP")
    fig = go.Figure()
    add_series(fig, sub, COLORS["systolic_bp"], "Systolic (mmHg)")
    add_pill_markers(fig, sub, COLORS["systolic_bp"])
    fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    sub = df[df["metric"]=="diastolic_bp"]
    st.subheader("Diastolic BP")
    fig = go.Figure()
    add_series(fig, sub, COLORS["diastolic_bp"], "Diastolic (mmHg)")
    add_pill_markers(fig, sub, COLORS["diastolic_bp"])
    fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    sub = df[df["metric"]=="spo2"]
    st.subheader("SpO₂")
    fig = go.Figure()
    add_series(fig, sub, COLORS["spo2"], "SpO₂ (%)")
    add_pill_markers(fig, sub, COLORS["spo2"])
    fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Combined overlay")
    fig = go.Figure()
    for key, label in [("pulse","Heart Rate"), ("systolic_bp","Systolic"), ("diastolic_bp","Diastolic"), ("spo2","SpO₂")]:
        sub = df[df["metric"]==key]
        add_series(fig, sub, COLORS[key], label)
    # pill markers for combined: put on pulse for reference (or compute per-series)
    sub_pulse = df[df["metric"]=="pulse"]
    add_pill_markers(fig, sub_pulse, "#111111")
    fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)