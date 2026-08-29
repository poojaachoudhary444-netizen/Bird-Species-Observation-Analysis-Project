"""
dashboard.py — Bird Species Observation Analysis: Streamlit Dashboard
=======================================================================
Interactive dashboard for exploring forest vs. grassland bird observations:
temporal trends, spatial/habitat patterns, species diversity, environmental
correlations, distance/behavior, observer trends, and conservation insights.

Run with:
    streamlit run dashboard.py

Reads from db/bird_observations.db (built by app.py). If it doesn't exist
yet, run `python app.py` first (or this dashboard will offer to run it).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bird Species Observation Analysis",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "bird_observations.db"
TABLE_NAME = "observations"

PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#F9A825"
HABITAT_COLORS = {"Forest": "#2E7D32", "Grassland": "#C0A22E"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _ensure_data_exists():
    """Self-healing bootstrap: if the DB (or even the raw workbooks) aren't
    present — e.g. on a fresh Streamlit Cloud deploy where only the .py files
    were committed — generate the sample data and run the ETL pipeline right
    here, once, so the app never shows a hard 'no data' error."""
    import app as etl  # local import to avoid slowing down normal reruns

    if DB_PATH.exists():
        return

    with st.spinner("First run: preparing data (this happens once)..."):
        if not etl.RAW_FOREST_PATH.exists() or not etl.RAW_GRASSLAND_PATH.exists():
            import generate_sample_data
            generate_sample_data.generate()
        etl.run_pipeline()


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    _ensure_data_exists()
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Flyover_Observed", "PIF_Watchlist_Status", "Regional_Stewardship_Status"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    return df


df_raw = load_data()

if df_raw.empty:
    st.title("🦅 Bird Species Observation Analysis")
    st.error(
        "Could not load or build the dataset automatically. If you're running "
        "locally, try:\n\n"
        "```bash\npython generate_sample_data.py\npython app.py\n```\n\n"
        "then rerun `streamlit run dashboard.py`."
    )
    st.stop()


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🦅 Filters")

habitat_opts = sorted(df_raw["Location_Type"].dropna().unique().tolist())
sel_habitat = st.sidebar.multiselect("Habitat Type", habitat_opts, default=habitat_opts)

unit_opts = sorted(df_raw["Admin_Unit_Code"].dropna().unique().tolist())
sel_units = st.sidebar.multiselect("Admin Unit", unit_opts, default=unit_opts)

year_opts = sorted(df_raw["Year"].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Year", year_opts, default=year_opts)

species_opts = sorted(df_raw["Common_Name"].dropna().unique().tolist())
sel_species = st.sidebar.multiselect(
    "Species (optional — leave blank for all)", species_opts, default=[]
)

observer_opts = sorted(df_raw["Observer"].dropna().unique().tolist())
sel_observers = st.sidebar.multiselect("Observer (optional)", observer_opts, default=[])

watchlist_only = st.sidebar.checkbox("PIF Watchlist species only", value=False)
stewardship_only = st.sidebar.checkbox("Regional Stewardship species only", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: NPS-style multi-unit bird monitoring survey "
    "(Forest & Grassland plots). Filters apply to every tab."
)

# Apply filters
df = df_raw[
    df_raw["Location_Type"].isin(sel_habitat)
    & df_raw["Admin_Unit_Code"].isin(sel_units)
    & df_raw["Year"].isin(sel_years)
]
if sel_species:
    df = df[df["Common_Name"].isin(sel_species)]
if sel_observers:
    df = df[df["Observer"].isin(sel_observers)]
if watchlist_only:
    df = df[df["PIF_Watchlist_Status"]]
if stewardship_only:
    df = df[df["Regional_Stewardship_Status"]]

if df.empty:
    st.warning("No observations match the current filters. Try widening your selection.")
    st.stop()


# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("🦅 Bird Species Observation Analysis")
st.caption("Forest vs. Grassland habitats · Temporal, spatial, species & conservation insights")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Observations", f"{len(df):,}")
k2.metric("Unique Species", f"{df['Common_Name'].nunique():,}")
k3.metric("Admin Units", f"{df['Admin_Unit_Code'].nunique()}")
k4.metric("Plots Surveyed", f"{df['Plot_Name'].nunique():,}")
k5.metric("Watchlist Species", f"{df.loc[df['PIF_Watchlist_Status'], 'Common_Name'].nunique()}")
watch_pct = (df["PIF_Watchlist_Status"].sum() / len(df) * 100) if len(df) else 0
k6.metric("Watchlist Obs. Share", f"{watch_pct:.1f}%")

st.markdown("---")

tabs = st.tabs([
    "📊 Overview",
    "🕒 Temporal Trends",
    "🗺️ Spatial / Habitat",
    "🐦 Species Analysis",
    "🌤️ Environment",
    "🔭 Distance & Behavior",
    "👤 Observers",
    "🛡️ Conservation",
])

# ==========================================================================
# TAB 1 — OVERVIEW
# ==========================================================================
with tabs[0]:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        habitat_counts = df["Location_Type"].value_counts().reset_index()
        habitat_counts.columns = ["Location_Type", "Observations"]
        fig = px.pie(
            habitat_counts, names="Location_Type", values="Observations", hole=0.5,
            color="Location_Type", color_discrete_map=HABITAT_COLORS,
            title="Observations by Habitat Type",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        unit_counts = df.groupby(["Admin_Unit_Code", "Location_Type"]).size().reset_index(name="Observations")
        fig = px.bar(
            unit_counts.sort_values("Observations", ascending=True),
            x="Observations", y="Admin_Unit_Code", color="Location_Type",
            color_discrete_map=HABITAT_COLORS, orientation="h",
            title="Observations by Admin Unit",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 Most-Observed Species")
    top_species = df["Common_Name"].value_counts().head(15).reset_index()
    top_species.columns = ["Common_Name", "Observations"]
    fig = px.bar(
        top_species.sort_values("Observations"), x="Observations", y="Common_Name",
        orientation="h", color="Observations", color_continuous_scale="Greens",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 2 — TEMPORAL TRENDS
# ==========================================================================
with tabs[1]:
    c1, c2 = st.columns(2)

    with c1:
        yearly = df.groupby(["Year", "Location_Type"]).size().reset_index(name="Observations")
        fig = px.line(
            yearly, x="Year", y="Observations", color="Location_Type", markers=True,
            color_discrete_map=HABITAT_COLORS, title="Observations by Year",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        seasonal = df.groupby(["Season", "Location_Type"]).size().reset_index(name="Observations")
        seasonal["Season"] = pd.Categorical(seasonal["Season"], categories=season_order, ordered=True)
        seasonal = seasonal.sort_values("Season")
        fig = px.bar(
            seasonal, x="Season", y="Observations", color="Location_Type", barmode="group",
            color_discrete_map=HABITAT_COLORS, title="Seasonal Distribution of Observations",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year × Month Observation Heatmap")
    heat = df.groupby(["Year", "Month_Name", "Month"]).size().reset_index(name="Observations")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heat_pivot = heat.pivot_table(index="Year", columns="Month_Name", values="Observations", fill_value=0)
    heat_pivot = heat_pivot.reindex(columns=[m for m in month_order if m in heat_pivot.columns])
    fig = px.imshow(
        heat_pivot, color_continuous_scale="Greens", aspect="auto",
        labels=dict(color="Observations"), title="Observation Density by Year & Month",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Observation Activity by Time of Day")
    time_bin_order = ["Early Morning (<7am)", "Morning (7-9am)", "Late Morning (9-11am)", "Midday+ (11am+)"]
    time_counts = df["Observation_Time_Bin"].value_counts().reindex(time_bin_order).reset_index()
    time_counts.columns = ["Time Window", "Observations"]
    fig = px.bar(time_counts, x="Time Window", y="Observations", color="Observations",
                 color_continuous_scale="Oranges")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 3 — SPATIAL / HABITAT
# ==========================================================================
with tabs[2]:
    c1, c2 = st.columns(2)

    with c1:
        div_by_unit = df.groupby(["Admin_Unit_Code", "Location_Type"])["Common_Name"].nunique().reset_index()
        div_by_unit.columns = ["Admin_Unit_Code", "Location_Type", "Unique_Species"]
        fig = px.bar(
            div_by_unit.sort_values("Unique_Species", ascending=True),
            x="Unique_Species", y="Admin_Unit_Code", color="Location_Type",
            color_discrete_map=HABITAT_COLORS, orientation="h",
            title="Species Diversity (Unique Species) by Admin Unit — Biodiversity Hotspots",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        plot_div = df.groupby("Plot_Name")["Common_Name"].nunique().reset_index()
        plot_div.columns = ["Plot_Name", "Unique_Species"]
        plot_div = plot_div.sort_values("Unique_Species", ascending=False).head(15)
        fig = px.bar(
            plot_div.sort_values("Unique_Species"), x="Unique_Species", y="Plot_Name",
            orientation="h", color="Unique_Species", color_continuous_scale="Teal",
            title="Top 15 Plots by Species Diversity",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Habitat Comparison: Observation Volume vs. Species Richness")
    comp = df.groupby("Location_Type").agg(
        Observations=("Common_Name", "size"),
        Unique_Species=("Common_Name", "nunique"),
        Plots=("Plot_Name", "nunique"),
    ).reset_index()
    fig = go.Figure()
    fig.add_bar(x=comp["Location_Type"], y=comp["Observations"], name="Observations",
                marker_color=ACCENT_COLOR, yaxis="y")
    fig.add_scatter(x=comp["Location_Type"], y=comp["Unique_Species"], name="Unique Species",
                     yaxis="y2", mode="markers+lines", marker=dict(size=14, color=PRIMARY_COLOR))
    fig.update_layout(
        yaxis=dict(title="Observations"),
        yaxis2=dict(title="Unique Species", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp, use_container_width=True, hide_index=True)

# ==========================================================================
# TAB 4 — SPECIES ANALYSIS
# ==========================================================================
with tabs[3]:
    c1, c2 = st.columns(2)

    with c1:
        sex_counts = df["Sex"].value_counts().reset_index()
        sex_counts.columns = ["Sex", "Observations"]
        fig = px.pie(sex_counts, names="Sex", values="Observations", hole=0.4,
                     title="Sex Ratio of Observed Birds")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        method_counts = df["ID_Method"].value_counts().reset_index()
        method_counts.columns = ["ID_Method", "Observations"]
        fig = px.bar(method_counts, x="ID_Method", y="Observations", color="ID_Method",
                     title="Identification Method Frequency")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Species Preference by Habitat (Top 12 species)")
    top12 = df["Common_Name"].value_counts().head(12).index
    pref = df[df["Common_Name"].isin(top12)].groupby(["Common_Name", "Location_Type"]).size().reset_index(name="Observations")
    fig = px.bar(
        pref, x="Common_Name", y="Observations", color="Location_Type", barmode="group",
        color_discrete_map=HABITAT_COLORS,
    )
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore a Species")
    chosen = st.selectbox("Pick a species", sorted(df["Common_Name"].unique()))
    sp_df = df[df["Common_Name"] == chosen]
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Observations", f"{len(sp_df):,}")
    sc2.metric("Admin Units Seen In", sp_df["Admin_Unit_Code"].nunique())
    sc3.metric("Avg. Distance (approx. m)", f"{sp_df['Distance_Approx_m'].mean():.0f}")
    sc4.metric("Watchlist Species?", "Yes" if sp_df["PIF_Watchlist_Status"].iloc[0] else "No")

# ==========================================================================
# TAB 5 — ENVIRONMENT
# ==========================================================================
with tabs[4]:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            df.sample(min(2000, len(df)), random_state=1),
            x="Temperature", y="Initial_Three_Min_Cnt", color="Location_Type",
            color_discrete_map=HABITAT_COLORS, opacity=0.5,
            title="Temperature vs. Initial 3-Min Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            df, x="Sky", y="Initial_Three_Min_Cnt", color="Location_Type",
            color_discrete_map=HABITAT_COLORS, title="Bird Counts by Sky Condition",
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            df.sample(min(2000, len(df)), random_state=1),
            x="Humidity", y="Distance_Approx_m", color="Location_Type",
            color_discrete_map=HABITAT_COLORS, opacity=0.5,
            title="Humidity vs. Observation Distance",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        dist_counts = df["Disturbance"].value_counts().reset_index()
        dist_counts.columns = ["Disturbance", "Observations"]
        fig = px.bar(dist_counts, x="Disturbance", y="Observations", color="Disturbance",
                     title="Effect of Disturbance on Observations")
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 6 — DISTANCE & BEHAVIOR
# ==========================================================================
with tabs[5]:
    c1, c2 = st.columns(2)

    with c1:
        dist_order = ["<= 50 Meters", "50 - 100 Meters", "> 100 Meters"]
        dc = df["Distance"].value_counts().reindex(dist_order).reset_index()
        dc.columns = ["Distance", "Observations"]
        fig = px.bar(dc, x="Distance", y="Observations", color="Observations",
                     color_continuous_scale="Blues", title="Observations by Distance Band")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        flyover = df["Flyover_Observed"].value_counts().rename({True: "Flyover", False: "Not Flyover"}).reset_index()
        flyover.columns = ["Flyover_Observed", "Observations"]
        fig = px.pie(flyover, names="Flyover_Observed", values="Observations", hole=0.4,
                     title="Flyover vs. Perched/Stationary Observations")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which Species Are Typically Seen Closer vs. Farther?")
    dist_by_species = (
        df.groupby("Common_Name")["Distance_Approx_m"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean")
        .reset_index()
    )
    fig = px.bar(
        dist_by_species.head(15), x="mean", y="Common_Name", orientation="h",
        color="mean", color_continuous_scale="Blues",
        labels={"mean": "Avg. Approx. Distance (m)"},
        title="Closest-Observed Species (min. 5 sightings)",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 7 — OBSERVERS
# ==========================================================================
with tabs[6]:
    c1, c2 = st.columns(2)

    with c1:
        obs_counts = df["Observer"].value_counts().reset_index()
        obs_counts.columns = ["Observer", "Observations"]
        fig = px.bar(obs_counts.sort_values("Observations"), x="Observations", y="Observer",
                     orientation="h", title="Observations Logged per Observer",
                     color="Observations", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        obs_div = df.groupby("Observer")["Common_Name"].nunique().reset_index()
        obs_div.columns = ["Observer", "Unique_Species_Reported"]
        fig = px.bar(
            obs_div.sort_values("Unique_Species_Reported"), x="Unique_Species_Reported", y="Observer",
            orientation="h", title="Unique Species Reported per Observer",
            color="Unique_Species_Reported", color_continuous_scale="Purples",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Do Repeat Visits Increase Species Count?")
    visit_div = df.groupby("Visit")["Common_Name"].nunique().reset_index()
    visit_div.columns = ["Visit #", "Unique_Species"]
    fig = px.line(visit_div, x="Visit #", y="Unique_Species", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# TAB 8 — CONSERVATION
# ==========================================================================
with tabs[7]:
    c1, c2 = st.columns(2)

    with c1:
        watch_by_unit = df[df["PIF_Watchlist_Status"]].groupby("Admin_Unit_Code")["Common_Name"].nunique().reset_index()
        watch_by_unit.columns = ["Admin_Unit_Code", "Watchlist_Species"]
        fig = px.bar(
            watch_by_unit.sort_values("Watchlist_Species", ascending=True),
            x="Watchlist_Species", y="Admin_Unit_Code", orientation="h",
            color="Watchlist_Species", color_continuous_scale="Reds",
            title="Watchlist Species Diversity by Admin Unit",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        steward = df.groupby(["Watchlist_Flag", "Location_Type"]).size().reset_index(name="Observations")
        fig = px.bar(
            steward, x="Watchlist_Flag", y="Observations", color="Location_Type",
            barmode="group", color_discrete_map=HABITAT_COLORS,
            title="At-Risk vs. Not-Listed Observations by Habitat",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Species Requiring Conservation Attention")
    watch_species = (
        df[df["PIF_Watchlist_Status"] | df["Regional_Stewardship_Status"]]
        .groupby(["Common_Name", "Scientific_Name"])
        .agg(
            Observations=("Common_Name", "size"),
            Admin_Units=("Admin_Unit_Code", "nunique"),
            PIF_Watchlist=("PIF_Watchlist_Status", "max"),
            Regional_Stewardship=("Regional_Stewardship_Status", "max"),
        )
        .reset_index()
        .sort_values("Observations", ascending=False)
    )
    st.dataframe(watch_species, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Built for the Bird Species Observation Analysis project · "
    "Data pipeline: app.py → SQLite → this dashboard. "
    "Sample data is synthetic until the real NPS dataset is loaded."
)
