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
from folium.raster_layers import ImageOverlay
from weather_reference import load_reference_temp, add_reference_to_chart
from osm_zones import enrich_with_zones, zone_summary_chart

try:
    from streamlit_folium import st_folium
except ModuleNotFoundError:
    st_folium = None

st.set_page_config(
    page_title="Sensordata op route",
    page_icon=None,
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Data laden & opschonen
# --------------------------------------------------------------------------- #
WERKELIJKE_START = {
    1: pd.Timestamp("2026-05-18 12:00:00"),
    2: pd.Timestamp("2026-05-27 12:00:00"),
    3: pd.Timestamp("2026-05-28 13:30:00"),
}

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


def idw_grid(lat, lon, values, n=160, power=2.0, mask_m=50.0):
    """Inverse-distance-weighted interpolatie naar een regelmatig raster.

    Schat de meetwaarde tussen de meetpunten in. Rastercellen die verder
    dan `mask_m` meter van het dichtstbijzijnde meetpunt liggen worden
    gemaskeerd (NaN), zodat de overlay alleen een band langs de gelopen
    route beslaat en niet de hele bounding box.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    values = np.asarray(values, dtype=float)

    ok = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(values))
    lat, lon, values = lat[ok], lon[ok], values[ok]
    if len(values) < 3:
        return None

    # Kleine marge zodat de band rond de route niet wordt afgekapt
    lat_pad = (lat.max() - lat.min()) * 0.05 or 1e-4
    lon_pad = (lon.max() - lon.min()) * 0.05 or 1e-4
    gy = np.linspace(lat.min() - lat_pad, lat.max() + lat_pad, n)
    gx = np.linspace(lon.min() - lon_pad, lon.max() + lon_pad, n)
    grid_x, grid_y = np.meshgrid(gx, gy)

    # Vectoriële IDW: afstand van elk rasterpunt tot elk meetpunt
    flat_y = grid_y.ravel()[:, None]
    flat_x = grid_x.ravel()[:, None]
    dist = np.sqrt((flat_y - lat[None, :]) ** 2
                   + (flat_x - lon[None, :]) ** 2)
    safe = np.where(dist < 1e-12, 1e-12, dist)
    weights = 1.0 / safe ** power
    grid = (weights @ values) / weights.sum(axis=1)
    grid = grid.reshape(grid_y.shape)

    # Afstand (in meters) tot het dichtstbijzijnde meetpunt per cel
    import math
    lat0 = float(lat.mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    dlat_m = (flat_y - lat[None, :]) * m_per_deg_lat
    dlon_m = (flat_x - lon[None, :]) * m_per_deg_lon
    nearest_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2).min(axis=1)
    nearest_m = nearest_m.reshape(grid_y.shape)

    # Maskeer alles buiten de band rond de route
    grid = np.where(nearest_m <= mask_m, grid, np.nan)

    bounds = [[float(gy.min()), float(gx.min())],
              [float(gy.max()), float(gx.max())]]
    return grid, bounds


def grid_to_rgba(grid, vmin, vmax):
    """Zet een waarde-raster om naar een RGBA-afbeelding (blauw→geel→rood).

    Gemaskeerde cellen (NaN) worden volledig transparant, zodat alleen de
    band langs de route gekleurd is.
    """
    masked = np.isnan(grid)
    if vmax == vmin:
        norm = np.zeros_like(grid)
    else:
        norm = np.clip((np.nan_to_num(grid) - vmin) / (vmax - vmin), 0, 1)

    r = np.where(norm < 0.5, norm / 0.5, 1.0)
    g = np.where(norm < 0.5, norm / 0.5, 1.0 - (norm - 0.5) / 0.5)
    b = np.where(norm < 0.5, 1.0 - norm / 0.5, 0.0)

    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (r * 255).astype(np.uint8)
    rgba[..., 1] = (g * 255).astype(np.uint8)
    rgba[..., 2] = (b * 255).astype(np.uint8)
    rgba[..., 3] = 165  # halftransparant zodat de kaart zichtbaar blijft
    rgba[masked, 3] = 0  # buiten de route-band: volledig transparant
    # ImageOverlay verwacht rij 0 = noordrand, np-grid heeft rij 0 = zuid
    return rgba[::-1]


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

    # Alleen rijen met geldige GPS-positie houden
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df = df[(df["latitude"].between(-90, 90)) &
            (df["longitude"].between(-180, 180))]

    # ------------------------------------------------------------------ #
    # Meetlopen detecteren op basis van tijdgaten > 1 uur
    # ------------------------------------------------------------------ #
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_gap"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["_run_raw"] = (df["_gap"] > 3600).cumsum()

    valid_runs = (
        df.groupby("_run_raw")["latitude"]
        .count()
        .loc[lambda s: s > 0]
        .index
    )
    df = df[df["_run_raw"].isin(valid_runs)].copy()

    run_map = {raw: i + 1 for i, raw in enumerate(sorted(valid_runs))}
    df["meting"] = df["_run_raw"].map(run_map)
    df = df.drop(columns=["_gap", "_run_raw"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Tijdscorrectie: verschuif elke meting naar de werkelijke starttijd
    # ------------------------------------------------------------------ #
    for meting_nr, werkelijke_start in WERKELIJKE_START.items():
        mask = df["meting"] == meting_nr
        if not mask.any():
            continue
        eerste_ts = df.loc[mask, "timestamp"].iloc[0]
        offset = werkelijke_start - eerste_ts
        df.loc[mask, "timestamp"] = df.loc[mask, "timestamp"] + offset

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Zone-classificatie: caching + persistente opslag op schijf
# --------------------------------------------------------------------------- #

ZONE_CACHE_FILE = Path(__file__).with_name("zones_cache.parquet")


def _zone_cache_key(df: pd.DataFrame) -> str:
    """Stabiele sleutel op basis van de GPS-coordinaten (op 6 decimalen)."""
    import hashlib

    coords = (
        df[["latitude", "longitude"]]
        .round(6)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.md5(coords).hexdigest()


@st.cache_data(show_spinner="Zones classificeren (eenmalig)...")
def classify_zones_cached(cache_key: str, file_bytes: bytes | None) -> pd.DataFrame:
    """Cache hangt alleen af van cache_key (string) — snel te hashen."""
    df = load_data(file_bytes)  # komt uit st.cache_data, dus gratis

    if ZONE_CACHE_FILE.exists():
        try:
            cached = pd.read_parquet(ZONE_CACHE_FILE)
            if cached["_cache_key"].iloc[0] == cache_key and "meting" in cached.columns:
                return cached.drop(columns=["_cache_key"])
        except Exception:
            pass

    enriched = enrich_with_zones(df)

    try:
        to_save = enriched.copy()
        to_save["_cache_key"] = cache_key
        to_save.to_parquet(ZONE_CACHE_FILE, index=False)
    except Exception:
        pass

    return enriched


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("Instellingen")

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
    ["Gekleurde route-punten", "Heatmap (interpolatie)", "Alleen lijn"],
)

if view_mode == "Heatmap (interpolatie)":
    mask_radius_m = st.sidebar.slider(
        "Breedte interpolatieband (m)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Alleen binnen deze afstand van de gelopen route wordt de "
             "geinterpoleerde waarde getoond.",
    )
else:
    mask_radius_m = 50

show_classification = st.sidebar.checkbox(
    "Toon classificatie per punt",
    value=False,
    help="Kleur de routepunten op omgevingstype en toon de classificatie "
         "in de popup van elk punt, plus een tabel per punt.",
)

# Tijdsfilter
# Classificeer de VOLLEDIGE dataset eenmalig (gecached + op schijf).
# Dit gebeurt voor het tijdsfilter, zodat het verschuiven van de slider
# of een rerun geen nieuwe (trage) classificatie triggert.
cols_before = set(df.columns)
cache_key = _zone_cache_key(df)
df = classify_zones_cached(cache_key, file_bytes)  # df ipv df als arg
meetlopen = sorted(df["meting"].unique())
gekozen_meting = st.sidebar.selectbox("Meetloop", meetlopen)

dff = df[df["meting"] == gekozen_meting].reset_index(drop=True)  # ← dit ontbrak

meetdagen = tuple(sorted(dff["timestamp"].dt.strftime("%Y%m%d").unique()))
reference_temp = load_reference_temp(dates=meetdagen)
if dff.empty:
    st.warning("Geen data voor de gekozen meetloop.")
    st.stop()



new_cols = [c for c in dff.columns if c not in cols_before]
zone_col = None
for candidate in ("zone", "zone_type", "zonetype", "omgevingstype",
                  "classification", "classificatie", "categorie", "category"):
    if candidate in dff.columns:
        zone_col = candidate
        break
if zone_col is None and new_cols:
    # val terug op de eerste door enrichment toegevoegde object-kolom
    obj_new = [c for c in new_cols if dff[c].dtype == object]
    zone_col = obj_new[0] if obj_new else new_cols[0]

# --------------------------------------------------------------------------- #
# Header & KPI's
# --------------------------------------------------------------------------- #

st.title("Sensordata over de gelopen route")
st.caption(
    "GPS-route met OpenStreetMap als ondergrond. De kleuren tonen de "
    + (f"classificatie (omgevingstype: **{zone_col}**)."
       if show_classification and zone_col
       else f"gekozen meetwaarde: **{metric_label}**.")
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


# Vaste, goed onderscheidbare kleurenpalet voor zone-classificatie
_ZONE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
zone_color_map: dict = {}
if zone_col is not None:
    unique_zones = [z for z in dff[zone_col].dropna().unique()]
    zone_color_map = {
        z: _ZONE_PALETTE[i % len(_ZONE_PALETTE)]
        for i, z in enumerate(sorted(unique_zones, key=str))
    }


def zone_color_for(value) -> str:
    if pd.isna(value):
        return "#999999"
    return zone_color_map.get(value, "#999999")


use_zone_colors = show_classification and zone_col is not None

if view_mode == "Gekleurde route-punten":
    for _, row in dff.iterrows():
        if use_zone_colors:
            point_color = zone_color_for(row[zone_col])
        else:
            point_color = color_for(row[metric_col])

        popup_html = f"<b>{row['timestamp']}</b><br>"
        if zone_col is not None:
            popup_html += f"Classificatie: <b>{row[zone_col]}</b><br>"
        popup_html += (
            f"Temp: {row['tempC']} °C<br>"
            f"CO₂: {row['co2_ppm']} ppm<br>"
            f"Vocht: {row['humidity']} %"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=point_color,
            fill=True,
            fill_color=point_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(fmap)

elif view_mode == "Heatmap (interpolatie)":
    interp = idw_grid(
        dff["latitude"].values,
        dff["longitude"].values,
        dff[metric_col].values,
        mask_m=mask_radius_m,
    )
    if interp is not None:
        grid, bounds = interp
        rgba = grid_to_rgba(grid, vmin, vmax)
        ImageOverlay(
            image=rgba,
            bounds=bounds,
            opacity=0.65,
            interactive=False,
            cross_origin=False,
            name=f"Interpolatie {metric_label}",
        ).add_to(fmap)
    else:
        st.warning("Te weinig punten voor interpolatie.")

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
if use_zone_colors:
    legend_items = " &nbsp; ".join(
        f"<span style='color:{c}'>● {z}</span>"
        for z, c in zone_color_map.items()
    )
    st.markdown(
        f"**Legenda — classificatie ({zone_col}):** {legend_items}",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"**Legenda — {metric_label}:** "
        f"<span style='color:#3388ff'>● {vmin:.1f} (laag)</span> &nbsp; "
        f"<span style='color:#ffff00;background:#444;padding:0 4px'>● midden</span> &nbsp; "
        f"<span style='color:#ff0000'>● {vmax:.1f} (hoog)</span>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Classificatie per punt (tabel) -- alleen tonen als de optie aanstaat
# --------------------------------------------------------------------------- #

if show_classification and zone_col is not None:
    st.subheader("Classificatie per punt")

    counts = (
        dff[zone_col]
        .value_counts(dropna=False)
        .rename_axis("Classificatie")
        .reset_index(name="Aantal punten")
    )
    counts["Aandeel"] = (
        counts["Aantal punten"] / counts["Aantal punten"].sum() * 100
    ).round(1).astype(str) + " %"
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.dataframe(
        dff[["timestamp", "latitude", "longitude", zone_col,
             "tempC", "co2_ppm", "humidity"]],
        use_container_width=True,
        hide_index=True,
    )
elif show_classification and zone_col is None:
    st.info(
        "Er is geen classificatiekolom gevonden in de data. "
        "Controleer of `enrich_with_zones` een zone-kolom toevoegt."
    )

# --------------------------------------------------------------------------- #
# Grafiek door de tijd
# --------------------------------------------------------------------------- #

st.subheader(f"{metric_label} door de tijd")
fig = px.line(
    dff,
    x="timestamp",
    y=metric_col,
    labels={"timestamp": "Tijd", metric_col: metric_label},
)
fig.update_traces(line_color="#e8590c")
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)

# Open-Meteo referentielijn toevoegen als temperatuur geselecteerd is
if metric_col == "tempC" and reference_temp is not None:
    fig = add_reference_to_chart(fig, dff, reference_temp)
    
st.plotly_chart(fig, use_container_width=True)
if metric_col == "tempC":
    from weather_reference import warmte_eiland_analyse
    warmte_eiland_analyse(dff, reference_temp)
with st.expander("Bekijk ruwe data"):
    cols = ["timestamp", "latitude", "longitude",
            "tempC", "co2_ppm", "humidity"]
    if zone_col is not None:
        cols.insert(3, zone_col)
    st.dataframe(dff[cols], use_container_width=True)

st.subheader("Sensorwaarden per omgevingstype")
zone_chart = zone_summary_chart(dff, metric_col, metric_label)
if zone_chart is not None:
    st.plotly_chart(zone_chart, use_container_width=True)

