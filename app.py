"""
Streamlit-app: Sensordata over de gelopen route (OpenStreetMap)
---------------------------------------------------------------
Toont temperatuur, CO2 en luchtvochtigheid langs de gelopen GPS-route.
GPS-coordinaten in de CSV staan in NMEA-formaat (DDMM.MMMMM) en worden
omgezet naar decimale graden.
"""

import io
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap
from knmi_reference import load_knmi_hourly, add_knmi_to_chart
from osm_zones import enrich_with_zones, zone_summary_chart

try:
    from streamlit_folium import st_folium
except ModuleNotFoundError:
    st_folium = None

st.set_page_config(
    page_title="Sensordata op route",
    page_icon="🗺️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Data laden & opschonen
# --------------------------------------------------------------------------- #


def nmea_to_decimal(value: pd.Series, hemisphere: pd.Series) -> pd.Series:
    """Zet NMEA DDMM.MMMM / DDDMM.MMMM om naar decimale graden.

    De graden zijn alle cijfers behalve de laatste twee voor de punt;
    de rest (minuten) wordt gedeeld door 60. Zuid/West worden negatief.
    """
    v = pd.to_numeric(value, errors="coerce")
    degrees = np.floor(v / 100.0)
    minutes = v - degrees * 100.0
    decimal = degrees + minutes / 60.0
    sign = hemisphere.astype(str).str.upper().map(
        {"N": 1, "E": 1, "S": -1, "W": -1})
    return decimal * sign


@st.cache_data
def load_data(file_bytes: bytes | None) -> pd.DataFrame:
    data_file = Path(__file__).with_name("DATA.CSV")
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(data_file)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # GPS omzetten
    df["latitude"] = nmea_to_decimal(df["lat"], df["ns"])
    df["longitude"] = nmea_to_decimal(df["lon"], df["ew"])

    # Numerieke meetwaarden
    for col in ["co2_ppm", "tempC", "humidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Alleen rijen met geldige GPS-positie houden voor de kaart
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df = df[(df["latitude"].between(-90, 90)) &
            (df["longitude"].between(-180, 180))]

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("⚙️ Instellingen")

uploaded = st.sidebar.file_uploader(
    "Eigen CSV uploaden (optioneel)", type=["csv", "CSV"]
)
file_bytes = uploaded.getvalue() if uploaded is not None else None

df = load_data(file_bytes)

if df.empty:
    st.error("Geen geldige GPS-rijen gevonden in de data.")
    st.stop()

METRICS = {
    "Temperatuur (°C)": "tempC",
    "CO₂ (ppm)": "co2_ppm",
    "Luchtvochtigheid (%)": "humidity",
}

metric_label = st.sidebar.selectbox("Meetwaarde", list(METRICS.keys()))
metric_col = METRICS[metric_label]

view_mode = st.sidebar.radio(
    "Weergave op kaart",
    ["Gekleurde route-punten", "Heatmap", "Alleen lijn"],
)

# Tijdsfilter
tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
if pd.notna(tmin) and pd.notna(tmax) and tmin != tmax:
    tijd_range = st.sidebar.slider(
        "Tijdsfilter",
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
        value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
    )
    mask = (df["timestamp"] >= tijd_range[0]) & (
        df["timestamp"] <= tijd_range[1])
    dff = df.loc[mask].reset_index(drop=True)
else:
    dff = df.copy()

if dff.empty:
    st.warning("Geen data in het gekozen tijdsinterval.")
    st.stop()
knmi = load_knmi_hourly(date="20260518", station=240)
dff = enrich_with_zones(dff)
# --------------------------------------------------------------------------- #
# Header & KPI's
# --------------------------------------------------------------------------- #

st.title("🗺️ Sensordata over de gelopen route")
st.caption(
    "GPS-route met OpenStreetMap als ondergrond. De kleuren tonen de "
    f"gekozen meetwaarde: **{metric_label}**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Meetpunten (met GPS)", f"{len(dff)}")
c2.metric(f"Gemiddeld {metric_label}", f"{dff[metric_col].mean():.1f}")
c3.metric("Minimum", f"{dff[metric_col].min():.1f}")
c4.metric("Maximum", f"{dff[metric_col].max():.1f}")

# --------------------------------------------------------------------------- #
# Kaart (Folium + OpenStreetMap)
# --------------------------------------------------------------------------- #

center = [dff["latitude"].mean(), dff["longitude"].mean()]
fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

# Route als lijn
coords = dff[["latitude", "longitude"]].values.tolist()
folium.PolyLine(
    coords, color="#3388ff", weight=3, opacity=0.6, tooltip="Gelopen route"
).add_to(fmap)

vmin, vmax = float(dff[metric_col].min()), float(dff[metric_col].max())


def color_for(value: float) -> str:
    """Kleurschaal van blauw (laag) via geel naar rood (hoog)."""
    if vmax == vmin or pd.isna(value):
        return "#3388ff"
    t = (value - vmin) / (vmax - vmin)
    if t < 0.5:
        # blauw -> geel
        r = int(255 * (t / 0.5))
        g = int(255 * (t / 0.5))
        b = int(255 * (1 - t / 0.5))
    else:
        # geel -> rood
        r = 255
        g = int(255 * (1 - (t - 0.5) / 0.5))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


if view_mode == "Gekleurde route-punten":
    for _, row in dff.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color_for(row[metric_col]),
            fill=True,
            fill_color=color_for(row[metric_col]),
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{row['timestamp']}</b><br>"
                f"Temp: {row['tempC']} °C<br>"
                f"CO₂: {row['co2_ppm']} ppm<br>"
                f"Vocht: {row['humidity']} %",
                max_width=220,
            ),
        ).add_to(fmap)

elif view_mode == "Heatmap":
    heat = dff[["latitude", "longitude", metric_col]].dropna().values.tolist()
    HeatMap(heat, radius=14, blur=10, min_opacity=0.3).add_to(fmap)

# Start- en eindmarker
folium.Marker(
    coords[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")
).add_to(fmap)
folium.Marker(
    coords[-1], tooltip="Einde", icon=folium.Icon(color="red", icon="stop")
).add_to(fmap)

if st_folium is not None:
    st_folium(fmap, width=None, height=560, returned_objects=[])
else:
    st.components.v1.html(fmap._repr_html_(), height=560)

# Legenda
st.markdown(
    f"**Legenda — {metric_label}:** "
    f"<span style='color:#3388ff'>● {vmin:.1f} (laag)</span> &nbsp; "
    f"<span style='color:#ffff00;background:#444;padding:0 4px'>● midden</span> &nbsp; "
    f"<span style='color:#ff0000'>● {vmax:.1f} (hoog)</span>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Grafiek door de tijd
# --------------------------------------------------------------------------- #

st.subheader(f"📈 {metric_label} door de tijd")
fig = px.line(
    dff,
    x="timestamp",
    y=metric_col,
    labels={"timestamp": "Tijd", metric_col: metric_label},
)
fig.update_traces(line_color="#e8590c")
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)

# KNMI referentielijn toevoegen als temperatuur geselecteerd is
if metric_col == "tempC" and knmi is not None:
    fig = add_knmi_to_chart(fig, dff, knmi)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Bekijk ruwe data"):
    st.dataframe(
        dff[
            [
                "timestamp",
                "latitude",
                "longitude",
                "tempC",
                "co2_ppm",
                "humidity",
            ]
        ],
        use_container_width=True,
    )
st.subheader("🏙️ Sensorwaarden per omgevingstype")
zone_chart = zone_summary_chart(dff, metric_col, metric_label)
if zone_chart is not None:
    st.plotly_chart(zone_chart, use_container_width=True)
