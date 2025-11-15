import os
from datetime import datetime, time as dtime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
import pytz

# ───────────────────────── Config
st.set_page_config(page_title="Quantaira Patient Detail", layout="wide")

# CSS
CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

CUSTOM_CSS = """
<style>
body, .stApp {
  background: #f1f5f9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #e5f3ef;
}
.sidebar-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
}
.breadcrumb {
  font-size: 13px;
  color: #6b7280;
  margin-bottom: 4px;
}

/* Time pills container */
.time-pill-container button {
  border-radius: 999px;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  color: #111827;
  font-weight: 600;
}
.time-pill-active button {
  background: linear-gradient(135deg,#6366f1,#22c55e);
  color: #ffffff;
  border-color: transparent;
  box-shadow: 0px 10px 25px rgba(99,102,241,0.25);
}

/* Stats card */
.stats-card {
  background: #ffffff;
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 12px 30px rgba(15,23,42,0.08);
  font-size: 14px;
}
.stats-card h4 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 16px;
}
.stats-card p {
  margin: 2px 0;
}

/* Section cards */
.section-card {
  background: #ffffff;
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 12px 30px rgba(15,23,42,0.06);
}

/* Recent meal card */
.meal-card {
  background: #f8fafc;
  border-radius: 16px;
  padding: 14px 18px;
  border: 1px solid #e5e7eb;
}
.meal-macros {
  display: flex;
  gap: 32px;
  margin-top: 8px;
}
.meal-macro-item span.label {
  display: block;
  font-size: 12px;
  color: #6b7280;
}
.meal-macro-item span.value {
  font-size: 18px;
  font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-render2.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY", "")

TIME_WINDOWS = {
    "24h": 24,
    "3d": 72,
    "7d": 168,
}

TZ_OPTIONS = [
    "UTC",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
]

METRIC_LABELS = {
    "pulse": "Heart Rate",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "spo2": "SpO₂",
}

METRIC_UNITS = {
    "pulse": "bpm",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "spo2": "%",
}

# LSL / USL per metric (example ranges)
LIMITS = {
    "pulse": (60.0, 100.0),
    "systolic_bp": (90.0, 120.0),
    "diastolic_bp": (60.0, 80.0),
    "spo2": (92.0, 100.0),
}

# Colors for segments
COL_RED = "#DC2626"
COL_YELLOW = "#FACC15"
COL_GREEN = "#22C55E"

# ───────────────────────── Helpers: measurements

def fetch_measurements(hours: int) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{API_BASE}/measurements",
            params={"hours": hours},
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc", "metric", "value_1", "value_2"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    mlow = df["metric"].astype(str).str.lower()

    # normalize metric names
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    bp_mask = mlow.isin({"blood_pressure", "bp"})
    if "value_2" in df.columns and bp_mask.any():
        bp = df[bp_mask].copy()
        sys = bp.assign(metric="systolic_bp", value=pd.to_numeric(bp["value_1"], errors="coerce"))
        dia = bp.assign(metric="diastolic_bp", value=pd.to_numeric(bp["value_2"], errors="coerce"))
        df = pd.concat([df[~bp_mask], sys, dia], ignore_index=True)
    else:
        df["value"] = pd.to_numeric(df.get("value_1"), errors="coerce")

    return df[["timestamp_utc", "metric", "value"]].dropna(subset=["timestamp_utc", "metric", "value"])


def to_local(series: pd.Series, tz_name: str) -> pd.Series:
    tz = pytz.timezone(tz_name)
    return series.dt.tz_convert(tz)


def compute_stats(sub: pd.DataFrame, metric: str) -> Dict[str, Any]:
    if sub.empty:
        return {}

    vals = sub["value"].astype(float)
    latest_row = sub.iloc[-1]
    lsl, usl = LIMITS.get(metric, (float("nan"), float("nan")))
    unit = METRIC_UNITS.get(metric, "")

    return {
        "lsl": lsl,
        "usl": usl,
        "latest": float(latest_row["value"]),
        "latest_time": latest_row["timestamp_local"],
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "unit": unit,
    }


def build_colored_segments(sub: pd.DataFrame, lsl: float, usl: float) -> Tuple[List[Dict], List[str]]:
    """
    Build segments and point colors for red/yellow/green ranges.
    Always returns (segments, point_colors) even if empty.
    """
    segments: List[Dict[str, Any]] = []
    point_colors: List[str] = []

    if sub.shape[0] == 0:
        return segments, point_colors

    ys = sub["value"].astype(float).to_numpy()
    xs = sub["timestamp_local"].to_numpy()

    # determine point colors first
    for y in ys:
        if y < lsl or y > usl:
            point_colors.append(COL_RED)
        elif lsl <= y <= usl:
            point_colors.append(COL_GREEN)
        else:
            point_colors.append(COL_YELLOW)

    if len(xs) < 2:
        # Not enough points for segments
        return segments, point_colors

    # segment color based on midpoint
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        mid = (y0 + y1) / 2.0

        if mid < lsl or mid > usl:
            col = COL_RED
        elif lsl <= mid <= usl:
            col = COL_GREEN
        else:
            col = COL_YELLOW

        segments.append(
            {
                "x": [x0, x1],
                "y": [y0, y1],
                "color": col,
            }
        )

    return segments, point_colors


def plot_metric_with_limits(
    df: pd.DataFrame,
    metric: str,
    tz_name: str,
    line_width: int,
    dot_size: int,
    show_limits: bool,
) -> Tuple[go.Figure, Dict[str, Any]]:
    sub = df[df["metric"] == metric].copy().sort_values("timestamp_utc")
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Time",
            yaxis_title=METRIC_UNITS.get(metric, ""),
        )
        return fig, {}

    sub["timestamp_local"] = to_local(sub["timestamp_utc"], tz_name)

    lsl, usl = LIMITS.get(metric, (float("nan"), float("nan")))
    segments, pt_colors = build_colored_segments(sub, lsl, usl)

    fig = go.Figure()

    # segments
    if segments:
        for seg in segments:
            fig.add_trace(
                go.Scatter(
                    x=seg["x"],
                    y=seg["y"],
                    mode="lines",
                    line=dict(width=line_width, color=seg["color"]),
                    hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
                    showlegend=False,
                )
            )
    else:
        # fallback single line
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp_local"],
                y=sub["value"],
                mode="lines",
                line=dict(width=line_width, color="#4b5563"),
                hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
                showlegend=False,
            )
        )

    # Points
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp_local"],
            y=sub["value"],
            mode="markers",
            marker=dict(
                size=dot_size,
                color=pt_colors if pt_colors else "#111827",
                line=dict(width=2, color="#ffffff"),
            ),
            hovertemplate="Value: %{y}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            showlegend=False,
        )
    )

    # LSL / USL dashed lines
    if show_limits and not np.isnan(lsl) and not np.isnan(usl):
        x0, x1 = sub["timestamp_local"].min(), sub["timestamp_local"].max()
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[lsl, lsl],
                mode="lines",
                line=dict(color="#9ca3af", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[usl, usl],
                mode="lines",
                line=dict(color="#9ca3af", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=10, b=40),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.25)",
        ),
        yaxis=dict(
            title=METRIC_UNITS.get(metric, ""),
            showgrid=True,
            gridcolor="rgba(226,232,240,0.6)",
        ),
    )

    stats = compute_stats(sub, metric)
    return fig, stats


def newest_timestamp_text(df: pd.DataFrame, tz_name: str) -> str:
    if df.empty:
        return "No data in source"
    latest = df["timestamp_utc"].max()
    tz = pytz.timezone(tz_name)
    latest_local = latest.tz_convert(tz)
    return f"Newest data point in source: {latest_local.strftime('%b %d, %H:%M %Z')}"

# ───────────────────────── Helpers: notes & meals (USDA)

def usda_search(query: str, api_key: str, page_size: int = 10) -> List[Dict[str, Any]]:
    if not api_key:
        st.warning("USDA_API_KEY is not set. Add it as an environment variable.")
        return []

    try:
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": query, "pageSize": page_size, "api_key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"USDA search failed: {e}")
        return []

    results = []
    for food in data.get("foods", []):
        desc = food.get("description", "Unknown food")
        nutrients = {n.get("nutrientName", ""): n.get("value", 0.0) for n in food.get("foodNutrients", [])}
        results.append(
            {
                "fdcId": food.get("fdcId"),
                "description": desc,
                "kcal": nutrients.get("Energy", 0.0),
                "protein": nutrients.get("Protein", 0.0),
                "carbs": nutrients.get("Carbohydrate, by difference", 0.0),
                "fat": nutrients.get("Total lipid (fat)", 0.0),
                "sodium": nutrients.get("Sodium, Na", 0.0),
            }
        )
    return results


def ensure_state():
    if "notes" not in st.session_state:
        st.session_state["notes"] = []
    if "meals" not in st.session_state:
        st.session_state["meals"] = []
    if "usda_results" not in st.session_state:
        st.session_state["usda_results"] = []
    if "time_choice" not in st.session_state:
        st.session_state["time_choice"] = "24h"

# ───────────────────────── Sidebar

ensure_state()

with st.sidebar:
    st.markdown('<div class="breadcrumb">Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Patient</div>', unsafe_allow_html=True)

    st.markdown("### Settings")

    tz_name = st.selectbox("Timezone", TZ_OPTIONS, index=0)

    line_width = st.slider("Line width", 1, 6, 4)
    dot_size = st.slider("Marker size (dots)", 6, 20, 10)

    show_limits = st.checkbox("Show LSL/USL dashed lines", True)

    st.markdown("### Limits mode")
    st.write("Using fixed LSL/USL per metric for now.")

# ───────────────────────── Main layout

st.markdown("## Patient Detail")

# Time window pills
time_cols = st.columns(len(TIME_WINDOWS))
for i, label in enumerate(TIME_WINDOWS.keys()):
    key = f"time_{label}"
    active = st.session_state["time_choice"] == label
    klass = "time-pill-container time-pill-active" if active else "time-pill-container"
    with time_cols[i]:
        st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
        if st.button(label, key=key):
            st.session_state["time_choice"] = label
        st.markdown("</div>", unsafe_allow_html=True)

time_label = st.session_state["time_choice"]
hours = TIME_WINDOWS[time_label]

df = fetch_measurements(hours)
if df.empty:
    st.warning("No data yet for this time window.")
    st.stop()

st.caption(newest_timestamp_text(df, tz_name))

# Metric tabs including systolic
tabs = st.tabs(["Heart Rate", "Systolic BP", "Diastolic BP", "SpO₂", "BP (both)"])

# Heart Rate
with tabs[0]:
    chart_col, stats_col = st.columns([3.2, 1.1])
    fig, stats = plot_metric_with_limits(df, "pulse", tz_name, line_width, dot_size, show_limits)
    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
    with stats_col:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("<h4>Stats</h4>", unsafe_allow_html=True)
        if stats:
            unit = stats["unit"]
            st.markdown(f"**LSL/USL:** {stats['lsl']:.1f} / {stats['usl']:.1f} {unit}")
            st.markdown(f"**Latest:** {stats['latest']:.1f} {unit}")
            st.markdown(
                f"Mean (µ): **{stats['mean']:.1f}**  \n"
                f"Std (σ): **{stats['std']:.1f}**"
            )
            st.markdown(f"Min: **{stats['min']:.1f} {unit}**")
            st.markdown(f"Max: **{stats['max']:.1f} {unit}**")
            st.markdown(
                f"Time: {stats['latest_time'].strftime('%Y-%m-%d %H:%M %Z')}"
            )
        else:
            st.markdown("No stats available.")
        st.markdown("</div>", unsafe_allow_html=True)

# Systolic
with tabs[1]:
    chart_col, stats_col = st.columns([3.2, 1.1])
    fig, stats = plot_metric_with_limits(df, "systolic_bp", tz_name, line_width, dot_size, show_limits)
    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
    with stats_col:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("<h4>Stats</h4>", unsafe_allow_html=True)
        if stats:
            unit = stats["unit"]
            st.markdown(f"**LSL/USL:** {stats['lsl']:.1f} / {stats['usl']:.1f} {unit}")
            st.markdown(f"**Latest:** {stats['latest']:.1f} {unit}")
            st.markdown(
                f"Mean (µ): **{stats['mean']:.1f}**  \n"
                f"Std (σ): **{stats['std']:.1f}**"
            )
            st.markdown(f"Min: **{stats['min']:.1f} {unit}**")
            st.markdown(f"Max: **{stats['max']:.1f} {unit}**")
            st.markdown(
                f"Time: {stats['latest_time'].strftime('%Y-%m-%d %H:%M %Z')}"
            )
        else:
            st.markdown("No stats available.")
        st.markdown("</div>", unsafe_allow_html=True)

# Diastolic
with tabs[2]:
    chart_col, stats_col = st.columns([3.2, 1.1])
    fig, stats = plot_metric_with_limits(df, "diastolic_bp", tz_name, line_width, dot_size, show_limits)
    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
    with stats_col:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("<h4>Stats</h4>", unsafe_allow_html=True)
        if stats:
            unit = stats["unit"]
            st.markdown(f"**LSL/USL:** {stats['lsl']:.1f} / {stats['usl']:.1f} {unit}")
            st.markdown(f"**Latest:** {stats['latest']:.1f} {unit}")
            st.markdown(
                f"Mean (µ): **{stats['mean']:.1f}**  \n"
                f"Std (σ): **{stats['std']:.1f}**"
            )
            st.markdown(f"Min: **{stats['min']:.1f} {unit}**")
            st.markdown(f"Max: **{stats['max']:.1f} {unit}**")
            st.markdown(
                f"Time: {stats['latest_time'].strftime('%Y-%m-%d %H:%M %Z')}"
            )
        else:
            st.markdown("No stats available.")
        st.markdown("</div>", unsafe_allow_html=True)

# SpO2
with tabs[3]:
    chart_col, stats_col = st.columns([3.2, 1.1])
    fig, stats = plot_metric_with_limits(df, "spo2", tz_name, line_width, dot_size, show_limits)
    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
    with stats_col:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("<h4>Stats</h4>", unsafe_allow_html=True)
        if stats:
            unit = stats["unit"]
            st.markdown(f"**LSL/USL:** {stats['lsl']:.1f} / {stats['usl']:.1f} {unit}")
            st.markdown(f"**Latest:** {stats['latest']:.1f} {unit}")
            st.markdown(
                f"Mean (µ): **{stats['mean']:.1f}**  \n"
                f"Std (σ): **{stats['std']:.1f}**"
            )
            st.markdown(f"Min: **{stats['min']:.1f} {unit}**")
            st.markdown(f"Max: **{stats['max']:.1f} {unit}**")
            st.markdown(
                f"Time: {stats['latest_time'].strftime('%Y-%m-%d %H:%M %Z')}"
            )
        else:
            st.markdown("No stats available.")
        st.markdown("</div>", unsafe_allow_html=True)

# BP both overlay
with tabs[4]:
    chart_col, stats_col = st.columns([3.2, 1.1])
    with chart_col:
        fig = go.Figure()
        for m in ["systolic_bp", "diastolic_bp"]:
            subfig, _ = plot_metric_with_limits(df, m, tz_name, line_width, dot_size, show_limits)
            for tr in subfig.data:
                fig.add_trace(tr)
        fig.update_layout(
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=40),
            hovermode="x unified",
            yaxis=dict(title="mmHg"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with stats_col:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("<h4>Stats</h4>", unsafe_allow_html=True)

        # systolic stats
        sub_sys = df[df["metric"] == "systolic_bp"].copy()
        if not sub_sys.empty:
            sub_sys["timestamp_local"] = to_local(sub_sys["timestamp_utc"], tz_name)
        systats = compute_stats(sub_sys, "systolic_bp") if not sub_sys.empty else {}

        # diastolic stats
        sub_dia = df[df["metric"] == "diastolic_bp"].copy()
        if not sub_dia.empty:
            sub_dia["timestamp_local"] = to_local(sub_dia["timestamp_utc"], tz_name)
        diastats = compute_stats(sub_dia, "diastolic_bp") if not sub_dia.empty else {}

        if systats:
            st.markdown(f"**Systolic LSL/USL:** {systats['lsl']:.1f} / {systats['usl']:.1f} mmHg")
            st.markdown(f"Latest: **{systats['latest']:.1f} mmHg**")
            st.markdown(f"Mean: **{systats['mean']:.1f}**, Std: **{systats['std']:.1f}**")
            st.markdown(f"Min: **{systats['min']:.1f}**, Max: **{systats['max']:.1f}**")
            st.markdown("---")
        if diastats:
            st.markdown(f"**Diastolic LSL/USL:** {diastats['lsl']:.1f} / {diastats['usl']:.1f} mmHg")
            st.markdown(f"Latest: **{diastats['latest']:.1f} mmHg**")
            st.markdown(f"Mean: **{diastats['mean']:.1f}**, Std: **{diastats['std']:.1f}**")
            st.markdown(f"Min: **{diastats['min']:.1f}**, Max: **{diastats['max']:.1f}**")
        if not systats and not diastats:
            st.markdown("No stats available.")
        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── Add Note & Add Meal

st.markdown("### Add Note & Add Meal (USDA)")

note_col, meal_col = st.columns(2)

# ----- Notes
with note_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📝 Add Note")
    note_text = st.text_area("Note", "", placeholder="e.g., felt dizzy after a walk", label_visibility="collapsed")
    use_now_note = st.checkbox("Use current time", value=True, key="note_use_now")

    if not use_now_note:
        note_date = st.date_input("When? (date)", datetime.now().date(), key="note_date")
        note_time = st.time_input("Time", datetime.now().time().replace(microsecond=0), key="note_time")
    else:
        note_date = datetime.now().date()
        note_time = datetime.now().time().replace(microsecond=0)

    if st.button("➕ Add Note"):
        if note_text.strip():
            dt = datetime.combine(note_date, note_time).replace(tzinfo=timezone.utc)
            st.session_state["notes"].append({"text": note_text.strip(), "time": dt})
            st.success("Note added.")
        else:
            st.warning("Please enter a note before adding.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Meals
with meal_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🍽️ Add Meal (USDA)")

    query = st.text_input("Search food (USDA)", "oatmeal", label_visibility="collapsed")
    use_now_meal = st.checkbox("Use current time", value=True, key="meal_use_now")

    if not use_now_meal:
        meal_date = st.date_input("When was it eaten? (date)", datetime.now().date(), key="meal_date")
        meal_time = st.time_input("Time", datetime.now().time().replace(microsecond=0), key="meal_time")
    else:
        meal_date = datetime.now().date()
        meal_time = datetime.now().time().replace(microsecond=0)

    if st.button("🔍 Search"):
        st.session_state["usda_results"] = usda_search(query, USDA_API_KEY)

    results = st.session_state.get("usda_results", [])
    if results:
        st.markdown("**Results**")
        for idx, food in enumerate(results):
            desc = food["description"]
            kcal = food["kcal"]
            p = food["protein"]
            c = food["carbs"]
            f = food["fat"]
            na = food["sodium"]

            st.markdown(
                f"**{desc}**  \n"
                f"{kcal:.0f} kcal • P {p:.1f} g • C {c:.1f} g • F {f:.1f} g • Na {na:.0f} mg"
            )
            if st.button("➕ Add", key=f"add_meal_{idx}"):
                dt = datetime.combine(meal_date, meal_time).replace(tzinfo=timezone.utc)
                st.session_state["meals"].append(
                    {
                        "description": desc,
                        "time": dt,
                        "kcal": kcal,
                        "protein": p,
                        "carbs": c,
                        "fat": f,
                        "sodium": na,
                    }
                )
                st.success("Meal added.")
    else:
        st.caption("Search above to see USDA results.")
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── Recent Meals

st.markdown("### Recent Meals")

meals = st.session_state.get("meals", [])
if not meals:
    st.caption("No meals added yet.")
else:
    meals_sorted = sorted(meals, key=lambda m: m["time"], reverse=True)[:3]
    for meal in meals_sorted:
        dt_local = meal["time"].astimezone(pytz.timezone(tz_name))
        st.markdown('<div class="meal-card">', unsafe_allow_html=True)
        st.markdown(
            f"**{meal['description']}**  \n"
            f"{dt_local.strftime('%Y-%m-%d %H:%M %Z')} • {meal['kcal']:.0f} kcal",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="meal-macros">'
            f'<div class="meal-macro-item"><span class="label">Protein</span><span class="value">{meal["protein"]:.1f} g</span></div>'
            f'<div class="meal-macro-item"><span class="label">Carbs</span><span class="value">{meal["carbs"]:.1f} g</span></div>'
            f'<div class="meal-macro-item"><span class="label">Fat</span><span class="value">{meal["fat"]:.1f} g</span></div>'
            f'<div class="meal-macro-item"><span class="label">Sodium</span><span class="value">{meal["sodium"]:.0f} mg</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
