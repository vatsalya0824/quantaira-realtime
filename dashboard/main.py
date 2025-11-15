import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ───────────────────────── CONFIG ─────────────────────────

st.set_page_config(page_title="Quantaira Dashboard", layout="wide")

CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-render2.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY", "")

COLORS = {
    "pulse": "#6F52ED",        # purple
    "systolic_bp": "#2D9CDB",  # blue
    "diastolic_bp": "#9B51E0", # violet
    "spo2": "#FF6B8A",         # pink
}

# default LSL/USL per metric
LIMITS = {
    "pulse": (60, 100),
    "systolic_bp": (90, 130),
    "diastolic_bp": (60, 85),
    "spo2": (92, 100),
}

# ───────────────────────── DATA HELPERS ─────────────────────────

def fetch_measurements(hours: int = 24) -> pd.DataFrame:
    """Fetch vitals from backend /measurements."""
    try:
        r = requests.get(f"{API_BASE}/measurements", params={"hours": hours}, timeout=20)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc", "metric", "value"])

    if not rows:
        return pd.DataFrame(columns=["created_utc", "metric", "value"])

    df = pd.DataFrame(rows)
    if "created_utc" not in df.columns:
        return pd.DataFrame(columns=["created_utc", "metric", "value"])

    df["timestamp_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    # split blood pressure into systolic / diastolic
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

    df = df[["timestamp_utc", "metric", "value"]].dropna()
    return df


def color_for_value(v: float, lsl: float, usl: float) -> str:
    if v < lsl:
        return "#F04438"  # red
    if v > usl:
        return "#F4B000"  # yellow
    return "#12B981"     # green


def build_colored_segments(
    sub: pd.DataFrame, lsl: float, usl: float
) -> Tuple[List[Tuple[List[pd.Timestamp], List[float], str]], List[str]]:
    """
    Build line segments where each segment has a single colour.
    Also returns per-point colours for markers.
    """
    xs = sub["ts_local"].tolist()
    ys = sub["value"].tolist()
    if len(xs) == 0:
        return [], []

    pt_colors = [color_for_value(v, lsl, usl) for v in ys]

    segments: List[Tuple[List[pd.Timestamp], List[float], str]] = []
    cur_color = pt_colors[0]
    seg_x = [xs[0]]
    seg_y = [ys[0]]

    for i in range(1, len(xs)):
        c = pt_colors[i]
        if c == cur_color:
            seg_x.append(xs[i])
            seg_y.append(ys[i])
        else:
            # close old segment
            if len(seg_x) > 1:
                segments.append((seg_x, seg_y, cur_color))
            # start new, duplicate previous point so line doesn't gap
            seg_x = [xs[i - 1], xs[i]]
            seg_y = [ys[i - 1], ys[i]]
            cur_color = c

    if len(seg_x) > 1:
        segments.append((seg_x, seg_y, cur_color))

    return segments, pt_colors


def compute_stats(sub: pd.DataFrame) -> Dict[str, float]:
    s = sub["value"]
    return {
        "latest": float(s.iloc[-1]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }

# ───────────────────────── USDA + MEALS ─────────────────────────

def usda_search(query: str, page_size: int = 10) -> List[dict]:
    if not USDA_API_KEY or not query.strip():
        return []
    try:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={
                "api_key": USDA_API_KEY,
                "query": query,
                "pageSize": page_size,
                "dataType": "Survey (FNDDS), SR Legacy",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("foods", [])
    except Exception as e:
        st.warning(f"USDA search failed: {e}")
        return []


def extract_macros(food: dict) -> dict:
    nutrients = {n.get("nutrientName", ""): n for n in food.get("foodNutrients", [])}

    def val(name, default=0.0):
        n = nutrients.get(name)
        if not n:
            return default
        return float(n.get("value") or 0.0)

    return {
        "kcals": val("Energy"),
        "protein": val("Protein"),
        "carbs": val("Carbohydrate, by difference"),
        "fat": val("Total lipid (fat)"),
        "sodium": val("Sodium, Na"),
    }


def ensure_state():
    if "recent_meals" not in st.session_state:
        st.session_state["recent_meals"] = []
    if "usda_results" not in st.session_state:
        st.session_state["usda_results"] = []

# ───────────────────────── UI HELPERS ─────────────────────────

def stats_card_html(stats: Dict[str, float], lsl: float, usl: float, newest: str) -> str:
    return f"""
    <div style="
        background:#ffffff;
        border-radius:18px;
        padding:18px 20px;
        box-shadow:0 12px 30px rgba(15,23,42,0.10);
        font-size:14px;
        ">
      <h4 style="margin:0 0 8px 0;font-weight:600;">Stats</h4>
      <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">
        Newest data point: {newest}
      </div>
      <div style="line-height:1.7;">
        <b>LSL/USL:</b> {lsl:.1f} / {usl:.1f}<br/>
        <b>Latest:</b> {stats["latest"]:.1f}<br/>
        μ <b>Mean:</b> {stats["mean"]:.1f}<br/>
        σ <b>Std:</b> {stats["std"]:.1f}<br/>
        <b>Min:</b> {stats["min"]:.1f}<br/>
        <b>Max:</b> {stats["max"]:.1f}
      </div>
    </div>
    """


def plot_metric_with_stats(
    df: pd.DataFrame,
    metric: str,
    tz: str,
    line_width: float,
    dot_size: float,
    show_limits: bool,
):
    sub = df[df["metric"] == metric].copy().sort_values("timestamp_utc")
    if sub.empty:
        st.warning(f"No data for {metric}")
        return

    lsl, usl = LIMITS.get(metric, (0.0, 1e6))
    sub["ts_local"] = sub["timestamp_utc"].dt.tz_convert(tz)

    segments, pt_colors = build_colored_segments(sub, lsl, usl)

    col_chart, col_stats = st.columns([3.4, 1.2])

    with col_chart:
        fig = go.Figure()

        # coloured line segments
        for xs, ys, color in segments:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # coloured dots
        fig.add_trace(
            go.Scatter(
                x=sub["ts_local"],
                y=sub["value"],
                mode="markers",
                marker=dict(
                    size=dot_size,
                    color=pt_colors,
                    line=dict(width=2, color="#111827"),
                ),
                hovertemplate="%{y:.1f}<br>%{x|%Y-%m-%d %H:%M %Z}<extra></extra>",
                showlegend=False,
            )
        )

        if show_limits:
            fig.add_hline(
                y=lsl,
                line_dash="dash",
                line_color="rgba(148,163,184,0.9)",
                annotation_text="LSL",
                annotation_position="top left",
            )
            fig.add_hline(
                y=usl,
                line_dash="dash",
                line_color="rgba(148,163,184,0.9)",
                annotation_text="USL",
                annotation_position="top left",
            )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=420,
            hovermode="x unified",
        )
        fig.update_xaxes(
            rangeslider=dict(visible=True),
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            griddash="dot",
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            griddash="dot",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        stats = compute_stats(sub)
        newest = sub["ts_local"].iloc[-1].strftime("%b %d, %H:%M %Z")
        st.markdown(stats_card_html(stats, lsl, usl, newest), unsafe_allow_html=True)


def plot_combined(df: pd.DataFrame, tz: str, normalize: bool):
    show_cols = ["pulse", "spo2", "systolic_bp", "diastolic_bp"]
    combo = df[df["metric"].isin(show_cols)].copy().sort_values("timestamp_utc")
    if combo.empty:
        st.warning("No data for combined view.")
        return

    combo["ts_local"] = combo["timestamp_utc"].dt.tz_convert(tz)

    if normalize:
        combo["value"] = combo.groupby("metric")["value"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
        )

    figc = px.line(
        combo,
        x="ts_local",
        y="value",
        color="metric",
        category_orders={"metric": show_cols},
        color_discrete_map=COLORS,
    )
    figc.update_traces(line=dict(width=2))
    figc.update_layout(
        margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        hovermode="x unified",
    )
    figc.update_xaxes(
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.2)",
        griddash="dot",
    )
    figc.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.2)",
        griddash="dot",
    )
    st.plotly_chart(figc, use_container_width=True)


def render_note_and_meals(tz: str):
    ensure_state()

    st.markdown("### Add Note & Add Meal (USDA)")

    col_note, col_meal = st.columns(2)

    # ----- Add Note -----
    with col_note:
        st.subheader("📝 Add Note")
        with st.form("note_form", clear_on_submit=True):
            note_text = st.text_area("Note", placeholder="e.g., felt dizzy after a walk")
            use_now = st.checkbox("Use current time", value=True)
            now = datetime.now(timezone.utc).astimezone()
            if use_now:
                note_dt = now
                st.markdown(
                    f"<div style='font-size:12px;color:#6b7280;'>Time: {now.strftime('%Y-%m-%d %H:%M %Z')}</div>",
                    unsafe_allow_html=True,
                )
            else:
                d = st.date_input("Date", now.date())
                t = st.time_input("Time", now.time())
                note_dt = datetime.combine(d, t).astimezone()
            submitted = st.form_submit_button("Add Note")
            if submitted and note_text.strip():
                st.success("Note captured (demo only, not persisted yet).")

    # ----- Add Meal -----
    with col_meal:
        st.subheader("🍽️ Add Meal (USDA)")
        with st.form("meal_form"):
            query = st.text_input("Search food (USDA)", value="oatmeal")
            use_now_meal = st.checkbox("Use current time", value=True, key="meal_use_now")
            now = datetime.now(timezone.utc).astimezone()
            if use_now_meal:
                meal_dt = now
                st.markdown(
                    f"<div style='font-size:12px;color:#6b7280;'>Eating time (local): {now.strftime('%Y-%m-%d %H:%M %Z')}</div>",
                    unsafe_allow_html=True,
                )
            else:
                d = st.date_input("When was it eaten? (date)", now.date(), key="meal_date")
                t = st.time_input("Time", now.time(), key="meal_time")
                meal_dt = datetime.combine(d, t).astimezone()

            search_clicked = st.form_submit_button("🔍 Search")
            if search_clicked:
                st.session_state["usda_results"] = usda_search(query)

        results = st.session_state.get("usda_results", [])
        for idx, food in enumerate(results):
            desc = food.get("description", "Unknown item")
            brand = food.get("brandName")
            label = f"{desc}" + (f" — {brand}" if brand else "")
            macros = extract_macros(food)
            kcals = macros["kcals"]
            protein = macros["protein"]
            carbs = macros["carbs"]
            fat = macros["fat"]
            sodium = macros["sodium"]

            with st.container():
                st.markdown(
                    f"**{label}**  \n"
                    f"{kcals:.0f} kcal · P {protein:.1f} g · C {carbs:.1f} g · "
                    f"F {fat:.1f} g · Na {sodium:.0f} mg"
                )
                if st.button("➕ Add", key=f"add_meal_{idx}"):
                    st.session_state["recent_meals"].append(
                        {
                            "label": label,
                            "time": meal_dt,
                            "kcals": kcals,
                            "protein": protein,
                            "carbs": carbs,
                            "fat": fat,
                            "sodium": sodium,
                        }
                    )
                    st.success("Meal added to Recent Meals.")

    # ----- Recent Meals -----
    st.markdown("### Recent Meals")
    meals = list(reversed(st.session_state["recent_meals"]))
    if not meals:
        st.info("No meals added yet.")
        return

    for m in meals[:5]:
        t_str = m["time"].strftime("%Y-%m-%d %H:%M %Z")
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border-radius:16px;
                padding:12px 16px;
                margin-bottom:10px;
                box-shadow:0 8px 20px rgba(15,23,42,0.06);
            ">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-weight:600;">{m["label"]}</div>
                  <div style="font-size:12px;color:#6b7280;">{t_str}</div>
                </div>
                <div style="font-size:12px;color:#4b5563;text-align:right;">
                  <div><b>{m["kcals"]:.0f}</b> kcal</div>
                  <div>Protein <b>{m["protein"]:.1f} g</b></div>
                  <div>Carbs <b>{m["carbs"]:.1f} g</b></div>
                  <div>Fat <b>{m["fat"]:.1f} g</b></div>
                  <div>Sodium <b>{m["sodium"]:.0f} mg</b></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ───────────────────────── MAIN ─────────────────────────

def main():
    ensure_state()

    # sidebar
    with st.sidebar:
        st.markdown("<h3>Settings</h3>", unsafe_allow_html=True)
        tz = st.selectbox(
            "Timezone",
            ["UTC", "US/Eastern", "US/Central", "US/Pacific"],
            index=0,
        )
        line_width = st.slider("Line width", 1, 6, 4)
        dot_size = st.slider("Marker size (dots)", 6, 20, 10)
        show_limits = st.checkbox("Show LSL/USL dashed lines", value=True)
        normalize_combined = st.checkbox("Normalize combined overlay", value=True)

    # top bar
    st.markdown(
        "<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>",
        unsafe_allow_html=True,
    )

    # time window pills in main area
    col_tw, _ = st.columns([1.3, 2.7])
    with col_tw:
        window_choice = st.radio(
            "Time window",
            options=["24h", "3 days", "7 days"],
            index=0,
            horizontal=True,
        )
    hours_map = {"24h": 24, "3 days": 72, "7 days": 168}
    hours = hours_map[window_choice]

    df = fetch_measurements(hours)
    if df.empty:
        st.warning("No data yet from devices for this time window.")
    else:
        tabs = st.tabs(
            ["❤️ Heart Rate", "💧 Systolic BP", "💜 Diastolic BP", "🫁 SpO₂", "📊 BP (both)"]
        )

        with tabs[0]:
            plot_metric_with_stats(df, "pulse", tz, line_width, dot_size, show_limits)
        with tabs[1]:
            plot_metric_with_stats(df, "systolic_bp", tz, line_width, dot_size, show_limits)
        with tabs[2]:
            plot_metric_with_stats(df, "diastolic_bp", tz, line_width, dot_size, show_limits)
        with tabs[3]:
            plot_metric_with_stats(df, "spo2", tz, line_width, dot_size, show_limits)
        with tabs[4]:
            plot_combined(df, tz, normalize_combined)

    st.markdown("---")
    render_note_and_meals(tz)


if __name__ == "__main__":
    main()
