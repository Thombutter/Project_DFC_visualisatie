"""
osm_zones.py  –  Omgevingstype per GPS-punt via OSM Nominatim
-------------------------------------------------------------
Koppelt elk meetpunt aan een omgevingstype (park, straat, woonwijk, etc.)
via reverse geocoding. Gebruikt een rasteraanpak om Nominatim-calls te
minimaliseren: punten die binnen dezelfde ~25m cel vallen delen één lookup.

Gebruik in app.py
-----------------
Stap 1 – import bovenaan:
    from osm_zones import enrich_with_zones, zone_summary_chart

Stap 2 – na je tijdsfilter (bijv. na knmi = load_knmi_hourly(...)):
    dff = enrich_with_zones(dff)

Stap 3 – voeg een nieuwe sectie toe na je bestaande grafiek:
    st.subheader("🏙️ Sensorwaarden per omgevingstype")
    zone_chart = zone_summary_chart(dff, metric_col, metric_label)
    if zone_chart is not None:
        st.plotly_chart(zone_chart, use_container_width=True)
"""

import time
from functools import lru_cache

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuratie
# --------------------------------------------------------------------------- #

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "sensordata-route-app/1.0"}

# Rastergrootte in graden (~25 m bij Amsterdam's breedtegraad)
GRID_SIZE = 0.0002

# Mapping van OSM-tags naar leesbare omgevingstypes
CATEGORY_MAP = {
    # Wegen
    "motorway": "Snelweg", "trunk": "Snelweg",
    "primary": "Hoofdweg", "secondary": "Hoofdweg",
    "tertiary": "Straat", "residential": "Woonstraat",
    "living_street": "Woonstraat", "service": "Woonstraat",
    "footway": "Voetpad", "path": "Voetpad",
    "cycleway": "Fietspad", "pedestrian": "Voetgangerszone",
    # Natuur & groen
    "park": "Park", "nature_reserve": "Natuur",
    "forest": "Bos", "wood": "Bos",
    "grass": "Groen", "meadow": "Groen",
    "garden": "Tuin/park",
    # Bebouwing
    "retail": "Winkelgebied", "commercial": "Commercieel",
    "industrial": "Industrieterrein", "construction": "Bouwterrein",
    "residential": "Woonwijk",
    # Water
    "water": "Water", "riverbank": "Water",
    # Overig
    "parking": "Parkeerplaats", "school": "School/campus",
    "university": "School/campus", "hospital": "Ziekenhuis",
    "station": "Treinstation", "platform": "OV-halte",
}

ZONE_COLORS = {
    "Snelweg":          "#e74c3c",
    "Hoofdweg":         "#e67e22",
    "Straat":           "#f39c12",
    "Woonstraat":       "#f1c40f",
    "Voetpad":          "#2ecc71",
    "Fietspad":         "#27ae60",
    "Voetgangerszone":  "#1abc9c",
    "Park":             "#16a085",
    "Natuur":           "#196F3D",
    "Bos":              "#145A32",
    "Groen":            "#58D68D",
    "Tuin/park":        "#82E0AA",
    "Woonwijk":         "#AED6F1",
    "Winkelgebied":     "#5DADE2",
    "Commercieel":      "#2E86C1",
    "Industrieterrein": "#7F8C8D",
    "Bouwterrein":      "#BDC3C7",
    "Water":            "#3498DB",
    "Parkeerplaats":    "#95A5A6",
    "School/campus":    "#9B59B6",
    "Treinstation":     "#8E44AD",
    "OV-halte":         "#A569BD",
    "Ziekenhuis":       "#EC407A",
    "Onbekend":         "#D5D8DC",
}

# --------------------------------------------------------------------------- #
# Nominatim lookup (gecached per rastercel)
# --------------------------------------------------------------------------- #

def _grid_cell(lat: float, lon: float) -> tuple:
    """Snap coördinaat naar rastercel."""
    return (round(lat / GRID_SIZE) * GRID_SIZE,
            round(lon / GRID_SIZE) * GRID_SIZE)


@lru_cache(maxsize=2048)
def _reverse_geocode(lat: float, lon: float) -> str:
    """Één Nominatim-call voor een rastercel. Resultaat gecached in geheugen."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 17,          # straatniveau
                "addressdetails": 0,
            },
            headers=HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()

        osm_type = data.get("type", "")
        osm_cat  = data.get("category", "")
        name     = data.get("name", "")

        # Probeer het type te mappen
        for key in [osm_type, osm_cat]:
            if key in CATEGORY_MAP:
                return CATEGORY_MAP[key]

        # Fallback op naam-hints
        name_l = name.lower()
        if any(w in name_l for w in ["park", "plantsoen", "bos"]):
            return "Park"
        if any(w in name_l for w in ["water", "gracht", "kanaal"]):
            return "Water"

        return "Onbekend"

    except Exception:
        return "Onbekend"


# --------------------------------------------------------------------------- #
# Hoofd-functie: verrijk DataFrame met zonetypes
# --------------------------------------------------------------------------- #

def enrich_with_zones(dff: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt kolom 'zone' toe aan dff via OSM reverse geocoding.

    Toont een Streamlit-voortgangsbalk tijdens het ophalen.
    Punten binnen dezelfde ~25m rastercel delen één API-call.
    """
    dff = dff.copy()

    # Bepaal unieke rastercellen
    dff["_cell"] = dff.apply(
        lambda r: _grid_cell(r["latitude"], r["longitude"]), axis=1
    )
    unique_cells = dff["_cell"].unique()
    n = len(unique_cells)

    # Haal al gecachede cellen op (geen nieuwe call nodig)
    cached = {c for c in unique_cells if _reverse_geocode.cache_info().currsize > 0
              and c in _reverse_geocode.__wrapped__.__code__.co_consts}

    new_cells = [c for c in unique_cells
                 if c not in getattr(_reverse_geocode, "_seen", set())]

    if not hasattr(_reverse_geocode, "_seen"):
        _reverse_geocode._seen = set()

    truly_new = [c for c in unique_cells if c not in _reverse_geocode._seen]
    n_new = len(truly_new)

    if n_new == 0:
        st.info(f"Alle {n} rastercellen al gecached — geen nieuwe API-calls nodig.")
    else:
        st.info(
            f"{n} unieke rastercellen gevonden uit {len(dff)} meetpunten. "
            f"~{n_new} Nominatim-calls nodig (~{n_new} seconden)."
        )

    cell_zone: dict[tuple, str] = {}
    bar = st.progress(0, text="Omgevingstypes ophalen…")

    for i, cell in enumerate(unique_cells):
        if cell not in cell_zone:
            cell_zone[cell] = _reverse_geocode(cell[0], cell[1])
            _reverse_geocode._seen.add(cell)
            if cell not in getattr(_reverse_geocode, "_seen_slept", set()):
                time.sleep(1.1)   # Nominatim rate limit: max 1 req/s
                if not hasattr(_reverse_geocode, "_seen_slept"):
                    _reverse_geocode._seen_slept = set()
                _reverse_geocode._seen_slept.add(cell)
        bar.progress((i + 1) / len(unique_cells),
                     text=f"Ophalen {i+1}/{len(unique_cells)}…")

    bar.empty()
    dff["zone"] = dff["_cell"].map(cell_zone)
    dff = dff.drop(columns=["_cell"])
    return dff


# --------------------------------------------------------------------------- #
# Samenvattingsgrafiek per omgevingstype
# --------------------------------------------------------------------------- #

def zone_summary_chart(
    dff: pd.DataFrame,
    metric_col: str,
    metric_label: str,
):
    """
    Geeft een Plotly-staafdiagram terug met gemiddelde meetwaarde per zone.
    Retourneert None als er geen 'zone'-kolom aanwezig is.
    """
    if "zone" not in dff.columns:
        return None

    summary = (
        dff.groupby("zone")[metric_col]
        .agg(gemiddelde="mean", n="count")
        .reset_index()
        .sort_values("gemiddelde", ascending=True)
    )

    colors = [ZONE_COLORS.get(z, "#D5D8DC") for z in summary["zone"]]

    fig = px.bar(
        summary,
        x="gemiddelde",
        y="zone",
        orientation="h",
        labels={"gemiddelde": f"Gemiddelde {metric_label}", "zone": "Omgevingstype"},
        text=summary["gemiddelde"].round(1),
        color="zone",
        color_discrete_map=ZONE_COLORS,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=40, t=10, b=10),
        height=max(250, len(summary) * 40),
        xaxis_title=metric_label,
        yaxis_title="",
    )
    return fig
