import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

APP_TITLE = "Crop Production & Yield Dashboard (India)"
APP_SUBTITLE = "Dashboard for state/district-level crop monitoring"

CROP_DATA_PATH = r"./data/India Agriculture Crop Production.csv"
GEOJSON_PATH = r"./data/india_state_geo.json"

@st.cache_data(show_spinner=True)
def load_crop_data():
    """Load and standardize crop production dataset."""
    df = pd.read_csv(CROP_DATA_PATH)

    df.columns = df.columns.str.strip().str.lower()
    col_yield = 'Yield'

    rename_map = {
        'State': "state",
        'District': "district",
        'Year': "year",
        'Season': "season",
        'Crop': "crop",
        'Area': "area",
        'Production': "production",
    }
    if col_yield:
        rename_map['Yield'] = "yield"

    df = df.rename(columns=rename_map)

    keep_cols = ["state", "district", "year", "season", "crop", "area", "production"]
    if "yield" in df.columns:
        keep_cols.append("yield")
    df = df[keep_cols]

    for col in ["state", "district", "season", "crop"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan})
        )
    df["year_str"] = df["year"].astype(str)
    df["year_numeric"] = (
        df["year_str"].str.extract(r"(\d{4})").astype(float)
    )
    mask_nan_year = df["year_numeric"].isna()
    if mask_nan_year.any():
        with pd.option_context("mode.use_inf_as_na", True):
            df.loc[mask_nan_year, "year_numeric"] = (
                pd.to_numeric(df.loc[mask_nan_year, "year_str"], errors="coerce")
            )

    for col in ["area", "production"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "yield" not in df.columns:
        df["yield"] = df["production"] / df["area"]
    else:
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
        missing_yield = df["yield"].isna()
        df.loc[missing_yield, "yield"] = (
            df.loc[missing_yield, "production"] / df.loc[missing_yield, "area"]
        )
    df = df.dropna(subset=["year_numeric", "area", "production", "yield"])

    df["state"] = df["state"].str.title()
    df["district"] = df["district"].str.title()

    return df


@st.cache_data(show_spinner=True)
def load_geojson():
    """Load Indian states GeoJSON and detect the state-name property key."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    sample_props = gj["features"][0]["properties"]
    candidates = ["NAME_1", "NAME"]
    state_key = None
    for k in candidates:
        if k in sample_props:
            state_key = k
            break
    if state_key is None:
        state_key = list(sample_props.keys())[0]

    return gj, state_key

def filter_data(df, crop, season, states, districts, years):
    filtered = df.copy()

    if crop is not None and len(crop) > 0:
        filtered = filtered[filtered["crop"].isin(crop)]

    if season is not None and len(season) > 0:
        filtered = filtered[filtered["season"].isin(season)]

    if states and len(states) > 0:
        filtered = filtered[filtered["state"].isin(states)]

    if districts and len(districts) > 0:
        filtered = filtered[filtered["district"].isin(districts)]

    if years and len(years) > 0:
        filtered = filtered[filtered["year_numeric"].isin(years)]

    return filtered


def compute_yield_trends(df_filtered):
    """Return time series summaries for area/production/yield by year."""
    ts = (
        df_filtered.groupby("year_numeric")[["area", "production", "yield"]]
        .sum()
        .reset_index()
        .sort_values("year_numeric")
    )
    return ts


def compute_state_avg_vs_region(df_all, df_region, crop, season, states, years):
    """Compare region yield vs state-wide (same crop/season) average."""
    years = sorted(years)
    if not years:
        return None

    base_filter = (df_all["crop"].isin(crop))
    if season:
        base_filter &= df_all["season"].isin(season)
    if states:
        base_filter &= df_all["state"].isin(states)

    df_base = df_all[base_filter]
    if df_base.empty:
        return None

    state_avg = (
        df_base.groupby("year_numeric")["yield"]
        .mean()
        .reset_index()
        .rename(columns={"yield": "state_avg_yield"})
    )

    region_avg = (
        df_region.groupby("year_numeric")["yield"]
        .mean()
        .reset_index()
        .rename(columns={"yield": "region_yield"})
    )

    compare = pd.merge(
        state_avg,
        region_avg,
        on="year_numeric",
        how="inner",
    ).sort_values("year_numeric")

    return compare


def compute_declining_states(df, crop, season, years):
    """
    Find states with >10% yield decline over last 5 years (or all available).
    """
    if not years:
        return pd.DataFrame(columns=["state", "start_year", "end_year", "start_yield", "end_yield", "pct_change"])

    years_sorted = sorted(years)
    if len(years_sorted) >= 5:
        window = years_sorted[-5:]
    else:
        window = years_sorted

    df_crop = df[df["crop"].isin(crop)].copy()
    if season:
        df_crop = df_crop[df_crop["season"].isin(season)]
    df_crop = df_crop[df_crop["year_numeric"].isin(window)]

    if df_crop.empty:
        return pd.DataFrame(columns=["state", "start_year", "end_year", "start_yield", "end_yield", "pct_change"])

    g = (
        df_crop.groupby(["state", "year_numeric"])["yield"]
        .mean()
        .reset_index()
    )

    results = []
    for state, group in g.groupby("state"):
        group = group.sort_values("year_numeric")
        start_year = group["year_numeric"].iloc[0]
        end_year = group["year_numeric"].iloc[-1]
        start_yield = group["yield"].iloc[0]
        end_yield = group["yield"].iloc[-1]
        if pd.isna(start_yield) or start_yield == 0:
            continue
        pct_change = (end_yield - start_yield) / start_yield * 100
        if pct_change <= -10:  
            results.append(
                {
                    "state": state,
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "start_yield": start_yield,
                    "end_yield": end_yield,
                    "pct_change": pct_change,
                }
            )

    if not results:
        return pd.DataFrame(columns=["state", "start_year", "end_year", "start_yield", "end_yield", "pct_change"])

    res_df = pd.DataFrame(results).sort_values("pct_change")
    return res_df


def generate_policy_brief(crop, season, years, states, ts_region, declining_states_table):
    """Create a short text-based policy brief summarizing key findings."""
    if ts_region is None or ts_region.empty:
        return "Not enough data to generate a policy brief for the selected filters."

    y_min = int(ts_region["year_numeric"].min())
    y_max = int(ts_region["year_numeric"].max())

    start_row = ts_region.iloc[0]
    end_row = ts_region.iloc[-1]

    start_yield = start_row["yield"]
    end_yield = end_row["yield"]
    if start_yield == 0:
        total_change_pct = np.nan
    else:
        total_change_pct = (end_yield - start_yield) / start_yield * 100

    states_str = ", ".join(states) if states else "All states"
    seasons_str = ", ".join(season) if isinstance(season, list) else season

    if declining_states_table is not None and not declining_states_table.empty:
        declining_list = ", ".join(declining_states_table["state"].tolist())
    else:
        declining_list = "None (no state shows >10% decline over the selected window)."

    brief = f"""
Crop Production & Yield Policy Brief

Crop: {", ".join(crop)}
Season(s): {seasons_str}
Years covered: {y_min}–{y_max}
Geography: {states_str}

1. Yield Trends

- Starting yield in {y_min}: {start_yield:,.2f} (tonnes/ha, approx.)
- Latest yield in {y_max}: {end_yield:,.2f} (tonnes/ha, approx.)
- Overall change in yield over the period: {total_change_pct:,.1f}% (approx.)

2. High-Risk States (Success Metric)

States with >10% yield decline over the last 5 years (or available window) for this crop and season:
- {declining_list}

3. Observations

- Yield trends should be interpreted alongside area and production trends.
- Sudden drops may indicate climatic shocks, pest/disease outbreaks, or policy / market disruptions.

4. Suggested Policy Actions

- Prioritize agronomic investigations and extension support in high-risk states.
- Target input subsidies, irrigation, and climate-resilient practices to districts with persistently low yields.
- Encourage data-driven monitoring by integrating this dashboard with field reports and remote-sensing indicators.

"""
    return brief.strip()


def show_correlation_matrix(df_filtered):
    corr_cols = ["area", "production", "yield"]
    corr = df_filtered[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(title="Correlation: Area vs Production vs Yield")
    st.plotly_chart(fig, use_container_width=True)


def show_time_series(ts):
    fig = px.line(
        ts,
        x="year_numeric",
        y=["area", "production", "yield"],
        markers=True,
    )
    fig.update_layout(
        title="Time Series: Area, Production, Yield",
        xaxis_title="Year",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_top_districts(df_filtered, top_n=10):
    by_district = (
        df_filtered.groupby("district")["production"]
        .sum()
        .reset_index()
        .sort_values("production", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        by_district,
        x="production",
        y="district",
        title=f"Top {top_n} Producing Districts (by Production)",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def show_yield_vs_area_scatter(df_filtered):
    fig = px.scatter(
        df_filtered,
        x="area",
        y="yield",
        color="state",
        hover_data=["district", "year_numeric"],
        title="Yield vs Production Area (by record)",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_cropwise_production(df_filtered):
    time_series_crop = df_filtered.groupby(['year_numeric', 'crop'])['production'].sum().unstack().fillna(0)
    fig = px.line(
        time_series_crop,
        x=time_series_crop.index,
        y=time_series_crop.columns,
        title="Crop-wise Production Time Series",
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Production")
    st.plotly_chart(fig, use_container_width=True)


def show_state_choropleth(df_filtered, geojson, state_key):
    if df_filtered.empty:
        st.info("No data available for the current filters to draw the map.")
        return

    by_state = (
        df_filtered.groupby("state")[["area", "production"]]
        .sum()
        .reset_index()
    )
    by_state["yield"] = by_state["production"] / by_state["area"]


    m = folium.Map(location=[22.5, 79], zoom_start=4, tiles="cartodbpositron")

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=by_state,
        columns=("state", "yield"),
        key_on=f"feature.properties.{state_key}",
        fill_opacity=0.7,
        line_opacity=0.8,
        highlight=True,
    ).add_to(m)

    folium.features.GeoJsonTooltip(
        fields=[state_key],
        aliases=["State:"],
    ).add_to(choropleth.geojson)

    st_folium(m, width=900, height=500)


def show_district_trend(df_filtered, selected_state):
    df_state = df_filtered[df_filtered["state"] == selected_state]
    if df_state.empty:
        st.info("No district-level data for the selected state with current filters.")
        return

    latest_year = df_state["year_numeric"].max()
    df_latest = df_state[df_state["year_numeric"] == latest_year]
    by_district = (
        df_latest.groupby("district")["production"]
        .sum()
        .reset_index()
        .sort_values("production", ascending=False)
    )

    fig_bar = px.bar(
        by_district,
        x="district",
        y="production",
        title=f"District-wise Production in {selected_state} (Year {int(latest_year)})",
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    ts_state = (
        df_state.groupby("year_numeric")[["area", "production", "yield"]]
        .sum()
        .reset_index()
        .sort_values("year_numeric")
    )

    fig_ts = px.line(
        ts_state,
        x="year_numeric",
        y=["area", "production", "yield"],
        markers=True,
        title=f"Time Series for {selected_state}",
    )
    st.plotly_chart(fig_ts, use_container_width=True)


def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if not Path(CROP_DATA_PATH).exists():
        st.error(f"Crop data file '{CROP_DATA_PATH}' not found. Place it in the same folder as app.py.")
        return
    if not Path(GEOJSON_PATH).exists():
        st.error(f"GeoJSON file '{GEOJSON_PATH}' not found. Place it in the same folder as app.py.")
        return

    df = load_crop_data()
    geojson, state_key = load_geojson()

    PAGES = {
        "Dashboard": render_dashboard_page,
        "Map View": render_map_view_page,
    }

    st.sidebar.header("Navigation")
    page_selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[page_selection]
    page(df, geojson, state_key)


def render_filters(df):
    with st.expander("Filters", expanded=True):
        st.markdown("Use the filters below to customize the dashboard and map views.")

        col1, col2 = st.columns(2)
        with col1:
            crops = sorted(df["crop"].dropna().unique())
            default_crops = ["Rice"] if "Rice" in crops else ([crops[0]] if crops else [])
            crop = st.multiselect("**Crop (multi-select)**", crops, default=default_crops)

            states = sorted(df["state"].dropna().unique())
            selected_states = st.multiselect("**State (multi-select)**", states, default=[])

            if selected_states:
                districts = sorted(df[df["state"].isin(selected_states)]["district"].dropna().unique())
            else:
                districts = sorted(df["district"].dropna().unique())
            selected_districts = st.multiselect("**District (multi-select)**", districts, default=[])

        with col2:
            seasons = sorted(df["season"].dropna().unique())
            season_default = seasons
            season = st.multiselect("**Season (multi-select)**", seasons, default=season_default)

            years = sorted(df["year_numeric"].dropna().unique().astype(int))
            selected_years = st.multiselect("**Year (multi-select)**", years, default=years)

    return crop, season, selected_states, selected_districts, selected_years


def render_dashboard_page(df, geojson, state_key):
    crop, season, selected_states, selected_districts, selected_years = render_filters(df)

    df_filtered = filter_data(df, crop, season, selected_states, selected_districts, selected_years)

    if df_filtered.empty:
        st.warning("No records found for the selected combination of filters.")
        return

  
    declining_states_table = compute_declining_states(df, crop, season, selected_years)
    st.subheader("Crop Production & Yield Dashboard")

    ts_region = compute_yield_trends(df_filtered)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Latest Year", int(ts_region["year_numeric"].max()))
    with col_m2:
        st.metric("Total Production (latest year)", f"{ts_region['production'].iloc[-1]:,.0f}")
    with col_m3:
        st.metric("Yield (latest year, t/ha approx.)", f"{ts_region['yield'].iloc[-1]:.2f}")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Correlation Matrix")
        show_correlation_matrix(df_filtered)

    with col2:
        st.markdown("### Yield vs State Average (time series)")
        compare = compute_state_avg_vs_region(df, df_filtered, crop, season, selected_states, selected_years)
        if compare is not None and not compare.empty:
            fig_comp = px.line(
                compare,
                x="year_numeric",
                y=["region_yield", "state_avg_yield"],
                markers=True,
                labels={"value": "Yield", "variable": "Series"},
                title="Region vs State Average Yield",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Not enough data to compare region yield against state average.")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Top Producing Districts")
        show_top_districts(df_filtered)
    with col4:
        st.markdown("### Yield Map (State level)")
        show_state_choropleth(df_filtered, geojson, state_key)


    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### Yield vs Production Area")
        show_yield_vs_area_scatter(df_filtered)
    with col6:
        st.markdown("### Crop-wise Production")
        show_cropwise_production(df_filtered)


    st.markdown("---")
    st.markdown("### States with >10% Yield Decline (Last 5 Years Window)")
    if declining_states_table is not None and not declining_states_table.empty:
        declining_states_table_display = declining_states_table.copy()
        declining_states_table_display["pct_change"] = declining_states_table_display["pct_change"].map(
            lambda x: f"{x:,.1f}%"
        )
        st.dataframe(declining_states_table_display, use_container_width=True)
    else:
        st.info("No states show a yield decline greater than 10% over the selected period.")


    st.markdown("---")
    st.markdown("### Export Policy Brief")
    brief = generate_policy_brief(crop, season, selected_years, selected_states, ts_region, declining_states_table)
    st.text_area("Preview (auto-generated brief)", brief, height=250)
    st.download_button(
        label="Download Policy Brief (.txt)",
        data=brief,
        file_name=f"policy_brief_{'_'.join(crop)}.txt",
        mime="text/plain",
    )


def render_map_view_page(df, geojson, state_key):
    crop, season, selected_states, selected_districts, selected_years = render_filters(df)

    df_filtered = filter_data(df, crop, season, selected_states, selected_districts, selected_years)

    if df_filtered.empty:
        st.warning("No records found for the selected combination of filters.")
        return
    st.subheader('State level Crop Yield Choropleth and District Trends')

    st.markdown(
        "Use this view to select a **State** on the left and see "
        "district-level production bars and trends for the same filters."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### State Map")
        show_state_choropleth(df_filtered, geojson, state_key)

    with col_right:
        st.markdown("### District-level View")
        state_options = sorted(df_filtered["state"].dropna().unique())
        default_state = state_options[0] if state_options else None
        selected_state_for_drill = st.selectbox(
            "Choose a state to inspect districts", state_options, index=0 if default_state else None
        )
        if selected_state_for_drill:
            show_district_trend(df_filtered, selected_state_for_drill)


if __name__ == "__main__":
    main()
