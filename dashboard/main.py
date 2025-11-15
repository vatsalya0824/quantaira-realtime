import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

# ───────────────────────── Setup ─────────────────────────

st.set_page_config(page_title="Quantaira Dashboard", layout="wide")

CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-render2.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY", "").strip()

METRIC_LABELS = {
    "pulse": "Heart Rate",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "spo2": "SpO₂",
}

# default clinical-ish limits (can tweak)
DEFAULT_LIMITS = {
    "pulse": (60.0, 100.0),
    "systolic_bp": (90.0, 130.0),
    "diastolic_bp": (60.0, 85.0),
    "spo2": (92.0, 100.0),
}

SEG_COLORS = {
    "low":   "#FF6B6B",   # red-ish
    "normal": "#00B894",  # green
    "high":  "#FFC107",   # yellow
}

# ───────────────────────── Helpers: data fetch ─────────────────────────

def fetch_measurements(hours: int) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/measurements", params={"hours": hours}, timeout=20)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc", "metric", "value_1", "value_2"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    # normalize metric names & values like last version
    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    bp_mask = mlow.isin({"blood_pressure", "bp"})
    if "value_2" in df.columns and bp_mask.any():
        bp = df[bp_mask].copy()
        sys = bp.assign(metric="systolic_bp",
                        value=pd.to_numeric(bp["value_1"], errors="coerce"))
        dia = bp.assign(metric="diastolic_bp",
                        value=pd.to_numeric(bp["value_2"], errors="coerce"))
        df = pd.concat([df[~bp_mask], sys, dia], ignore_index=True)
    else:
        df["value"] = pd.to_numeric(df.get("value_1"), errors="coerce")

    return df[["created_utc", "metric", "value"]].dropna(subset=["created_utc", "metric", "value"])


def to_timezone(series: pd.Series, tz_name: str) -> pd.Series:
    # series is timezone-aware UTC
    try:
        if tz_name == "UTC":
            return series.dt.tz_convert("UTC")
        return series.dt.tz_convert(ZoneInfo(tz_name))
    except Exception:
        return series  # fallback to UTC if something is weird


# ───────────────────────── Helpers: colored segments & stats ─────────────────────────

def classify_value(v: float, lsl: float, usl: float) -> str:
    if v < lsl:
        return "low"
    if v > usl:
        return "high"
    return "normal"


def build_colored_segments(sub: pd.DataFrame, lsl: float, usl: float
                           ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      segments: list of {x: [...], y: [...], color: hex}
      point_colors: list of colors for each data point (markers)
    Never raises ValueError even for 0 or 1 points.
    """
    if sub.empty:
        return [], []

    xs = sub["ts_local"].tolist()
    ys = sub["value"].tolist()

    point_colors = [SEG_COLORS[classify_value(v, lsl, usl)] for v in ys]

    segments: List[Dict[str, Any]] = []
    if len(xs) >= 2:
        for i in range(len(xs) - 1):
            v1, v2 = ys[i], ys[i + 1]
            # color segment by "worst" status between the two endpoints
            status1 = classify_value(v1, lsl, usl)
            status2 = classify_value(v2, lsl, usl)
            status = status1 if status1 != "normal" else status2
            if status == "normal" and status2 != "normal":
                status = status2
            color = SEG_COLORS[status]
            segments.append({
                "x": [xs[i], xs[i + 1]],
                "y": [ys[i], ys[i + 1]],
                "color": color,
            })

    return segments, point_colors


def render_stats_card(sub: pd.DataFrame, lsl: float, usl: float):
    latest_row = sub.iloc[-1]
    latest_val = float(latest_row["value"])
    latest_ts = latest_row["ts_local"]

    mu = float(sub["value"].mean())
    sigma = float(sub["value"].std(ddof=0))
    vmin = float(sub["value"].min())
    vmax = float(sub["value"].max())

    st.markdown(
        f"""
        <div class="stats-card">
          <div class="stats-title">Stats</div>
          <div>LSL/USL: <b>{lsl:.1f}</b> / <b>{usl:.1f}</b></div>
          <div>Latest: <b>{latest_val:.1f}</b> at {latest_ts:%b %d, %H:%M}</div>
          <div>μ Mean: <b>{mu:.1f}</b></div>
          <div>σ Std: <b>{sigma:.1f}</b></div>
          <div>Min: <b>{vmin:.1f}</b></div>
          <div>Max: <b>{vmax:.1f}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────── USDA helpers ─────────────────────────

def search_usda_foods(query: str) -> list[dict]:
    if not USDA_API_KEY:
        st.info("USDA_API_KEY is not set; cannot search USDA foods.")
        return []
    if not query.strip():
        return []

    try:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"api_key": USDA_API_KEY, "query": query, "pageSize": 10},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("foods", [])
    except Exception as e:
        st.error(f"USDA search failed: {e}")
        return []


def extract_macros(food: dict) -> dict:
    nutrients = {n.get("nutrientName", ""): n for n in food.get("foodNutrients", [])}

    def get(n_name: str, fallback: float = 0.0) -> float:
        for key, n in nutrients.items():
            if n_name.lower() in key.lower():
                try:
                    return float(n.get("value", fallback))
                except Exception:
                    return fallback
        return fallback

    return {
        "kcal": get("Energy"),
        "protein": get("Protein"),
        "carbs": get("Carbohydrate"),
        "fat": get("Total lipid"),
        "sodium": get("Sodium"),
    }


def format_usda_result(food: dict) -> str:
    name = food.get("description", "Unknown").title()
    brand = food.get("brandOwner") or ""
    macros = extract_macros(food)

    subtitle = f"{int(macros['kcal'])} kcal · P {macros['protein']:.1f}g · C {macros['carbs']:.1f}g · F {macros['fat']:.1f}g · Na {macros['sodium']:.0f}mg"
    if brand:
        name = f"{name} — {brand}"

    return f"""
    <div class="meal-result">
        <div class="meal-name">{name}</div>
        <div class="meal-subtitle">{subtitle}</div>
    </div>
    """


def add_meal_to_backend(food: dict, eaten_at: datetime):
    payload = {
        "timestamp": eaten_at.isoformat(),
        "food": food,
        "macros": extract_macros(food),
    }
    try:
        r = requests.post(f"{API_BASE}/meals", json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        # don't crash UI; just show toast
        st.warning(f"Meal saved locally (backend POST failed: {e})")


def add_note_to_backend(text: str, at_time: datetime):
    payload = {"timestamp": at_time.isoformat(), "text": text}
    try:
        r = requests.post(f"{API_BASE}/notes", json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        st.warning(f"Note saved locally (backend POST failed: {e})")


def fetch_recent_meals(hours: int) -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/meals", params={"hours": hours}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        # it's okay if this endpoint doesn't exist yet
        return []


# ───────────────────────── Plotting ─────────────────────────

def plot_metric_with_limits(
    df: pd.DataFrame,
    metric: str,
    tz_name: str,
    line_width: int,
    dot_size: int,
    show_limits: bool,
):
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        st.warning(f"No data for {METRIC_LABELS.get(metric, metric)}")
        return

    # timezone conversion
    sub["ts_local"] = to_timezone(sub["created_utc"], tz_name)
    sub = sub.sort_values("ts_local")

    # choose limits: default map or derive from data
    if metric in DEFAULT_LIMITS:
        lsl, usl = DEFAULT_LIMITS[metric]
    else:
        mu = sub["value"].mean()
        sigma = sub["value"].std(ddof=0) or 1.0
        lsl, usl = float(mu - sigma), float(mu + sigma)

    segments, point_colors = build_colored_segments(sub, lsl, usl)

    fig = go.Figure()

    # colored line segments
    for seg in segments:
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode="lines",
                line=dict(width=line_width, color=seg["color"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # markers on top (black outline)
    fig.add_trace(
        go.Scatter(
            x=sub["ts_local"],
            y=sub["value"],
            mode="markers",
            marker=dict(size=dot_size, color=point_colors,
                        line=dict(width=1, color="#000000")),
            hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M %Z}<extra></extra>",
            showlegend=False,
        )
    )

    if show_limits:
        fig.add_hline(
            y=lsl,
            line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"),
        )
        fig.add_hline(
            y=usl,
            line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"),
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    left, right = st.columns([3, 1.1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        render_stats_card(sub, lsl, usl)


# ───────────────────────── UI ─────────────────────────

with st.sidebar:
    st.markdown("### Home")
    st.button("Patient", disabled=True)

    st.markdown("### Settings")

    tz_choice = st.selectbox(
        "Timezone",
        ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific"],
        index=0,
    )

    time_window_label = st.radio(
        "Time window",
        ["24 hr", "3 days", "7 days"],
        index=0,
    )
    HOURS_MAP = {"24 hr": 24, "3 days": 72, "7 days": 168}
    hours = HOURS_MAP[time_window_label]

    line_width = st.slider("Line width", 1, 6, 4)
    dot_size = st.slider("Marker size (dots)", 6, 20, 10)
    show_limits = st.checkbox("Show LSL/USL dashed lines", value=True)

    st.markdown("### Limits mode")
    st.caption("Using default clinical ranges per metric (you can tweak in code).")

st.markdown(
    "<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>",
    unsafe_allow_html=True,
)

df = fetch_measurements(hours)
if df.empty:
    st.warning("No vital data yet.")
    st.stop()

newest_ts = df["created_utc"].max()
st.caption(f"Newest data point in source: {newest_ts.astimezone(timezone.utc):%b %d, %H:%M UTC}")

tabs = st.tabs(["Heart Rate", "Systolic BP", "Diastolic BP", "SpO₂", "BP (both)"])

with tabs[0]:
    plot_metric_with_limits(df, "pulse", tz_choice, line_width, dot_size, show_limits)

with tabs[1]:
    plot_metric_with_limits(df, "systolic_bp", tz_choice, line_width, dot_size, show_limits)

with tabs[2]:
    plot_metric_with_limits(df, "diastolic_bp", tz_choice, line_width, dot_size, show_limits)

with tabs[3]:
    plot_metric_with_limits(df, "spo2", tz_choice, line_width, dot_size, show_limits)

with tabs[4]:
    # simple combo view reusing single-metric plots side-by-side if you like
    st.info("Combined BP overlay is coming soon – currently using individual tabs.")

st.markdown("---")

# ───────────── Add Note & Add Meal (USDA) ─────────────

col_note, col_meal = st.columns(2)

# ----- Add Note -----
with col_note:
    st.subheader("📝 Add Note")

    note_text = st.text_area("Note", placeholder="e.g., felt dizzy after a walk")

    use_now_note = st.checkbox("Use current time", value=True, key="note_use_now")
    if use_now_note:
        note_dt = datetime.now(timezone.utc)
    else:
        c1, c2 = st.columns(2)
        with c1:
            note_date = st.date_input(
                "When? (date)",
                value=datetime.now(timezone.utc).date(),
                key="note_date",
            )
        with c2:
            note_time = st.time_input(
                "Time (UTC)",
                value=datetime.now(timezone.utc).time().replace(second=0, microsecond=0),
                key="note_time",
            )
        note_dt = datetime.combine(note_date, note_time).replace(tzinfo=timezone.utc)

    st.caption(f"Note time (UTC): {note_dt.strftime('%Y-%m-%d %H:%M')}")

    if st.button("➕ Add Note"):
        if not note_text.strip():
            st.error("Please write a note first.")
        else:
            add_note_to_backend(note_text.strip(), note_dt)
            st.success("Note saved.")

# ----- Add Meal (USDA) -----
with col_meal:
    st.subheader("🍽️ Add Meal (USDA)")

    query = st.text_input("Search food (USDA)", key="meal_query", value="oatmeal")

    use_now_meal = st.checkbox("Use current time", value=True, key="meal_use_now")
    if use_now_meal:
        eating_dt = datetime.now(timezone.utc)
    else:
        c1, c2 = st.columns(2)
        with c1:
            eat_date = st.date_input(
                "When was it eaten? (date)",
                value=datetime.now(timezone.utc).date(),
                key="meal_date",
            )
        with c2:
            eat_time = st.time_input(
                "Time (UTC)",
                value=datetime.now(timezone.utc).time().replace(second=0, microsecond=0),
                key="meal_time",
            )
        eating_dt = datetime.combine(eat_date, eat_time).replace(tzinfo=timezone.utc)

    st.caption(f"Eating time (UTC): {eating_dt.strftime('%Y-%m-%d %H:%M')}")

    if st.button("🔍 Search", key="meal_search"):
        results = search_usda_foods(query)
        st.session_state["meal_search_results"] = (results, eating_dt)

    results_block = st.session_state.get("meal_search_results")
    if results_block:
        results, eaten_at_for_results = results_block
        if not results:
            st.info("No USDA results.")
        else:
            for food in results:
                st.markdown(format_usda_result(food), unsafe_allow_html=True)
                if st.button("➕ Add", key=f"add_meal_{food.get('fdcId')}"):
                    add_meal_to_backend(food, eaten_at_for_results)
                    st.success("Meal added.")

st.markdown("---")

# ───────────── Recent Meals ─────────────

st.subheader("🍽️ Recent Meals")

recent_meals = fetch_recent_meals(hours)
if not recent_meals:
    st.info("No meals recorded yet.")
else:
    for m in recent_meals:
        when = datetime.fromisoformat(m["timestamp"]).astimezone(timezone.utc)
        macros = m.get("macros", {})
        kcal = macros.get("kcal", 0)
        prot = macros.get("protein", 0.0)
        carbs = macros.get("carbs", 0.0)
        fat = macros.get("fat", 0.0)
        sodium = macros.get("sodium", 0.0)
        name = m.get("name") or m.get("food", {}).get("description", "Meal")

        st.markdown(
            f"""
            <div class="recent-meal-card">
              <div class="meal-header">
                <span class="meal-title">{name}</span>
                <span class="meal-kcal">{int(kcal)} kcal</span>
              </div>
              <div class="meal-meta">{when:%Y-%m-%d %H:%M} UTC</div>
              <div class="meal-macros">
                <span>Protein {prot:.1f} g</span>
                <span>Carbs {carbs:.1f} g</span>
                <span>Fat {fat:.1f} g</span>
                <span>Sodium {sodium:.0f} mg</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
