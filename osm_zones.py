"""
osm_zones.py
------------
Classificeert elk GPS-meetpunt naar het type weg waarop het ligt, op
basis van OpenStreetMap-data (Overpass API).

Er worden uitsluitend drie categorieen gebruikt:

    - "Voetpad"
    - "Fietspad"
    - "Hoofdweg"

Werkwijze
=========
1. Eenmalig worden alle relevante wegen (highways) binnen de bounding box
   van de route uit OSM opgehaald via de Overpass API.
2. Voor elk meetpunt wordt de dichtstbijzijnde weg gezocht. De OSM
   `highway`-tag van die weg wordt naar een van de drie categorieen
   gemapt.
3. Ligt er geen bruikbare weg binnen `MAX_SNAP_M` meter, of valt de tag
   buiten de drie categorieen, dan krijgt het punt de classificatie van
   het dichtstbijzijnde punt dat WEL een van de drie categorieen heeft
   (nearest-neighbour fallback).

De resulterende kolom heet ``classificatie``.

Als de Overpass API niet bereikbaar is, valt de module terug op een
geometrische heuristiek zodat de app blijft werken (de kolom wordt dan
nog steeds gevuld, met een waarschuwing in de Streamlit-UI).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

try:  # plotly is optioneel voor de samenvattingsgrafiek
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

try:  # streamlit alleen gebruikt voor nette waarschuwingen
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# --------------------------------------------------------------------------- #
# Categorieen
# --------------------------------------------------------------------------- #

VOETPAD = "Voetpad"
FIETSPAD = "Fietspad"
WOONSTRAAT = "Woonstraat"
HOOFDWEG = "Hoofdweg"
ONBEKEND = "Onbekend"  # alleen gebruikt als OSM onbereikbaar is

CATEGORIES = (VOETPAD, FIETSPAD, WOONSTRAAT, HOOFDWEG)

# OSM highway-tag  ->  onze categorie
HIGHWAY_MAP = {
    # Voetpad
    "footway": VOETPAD, "pedestrian": VOETPAD, "path": VOETPAD,
    "steps": VOETPAD, "track": VOETPAD, "bridleway": VOETPAD,
    # Fietspad
    "cycleway": FIETSPAD,
    # Woonstraat (lokale straten met gemengd verkeer; GEEN doorgaande weg)
    "living_street": WOONSTRAAT, "residential": WOONSTRAAT,
    "service": WOONSTRAAT, "unclassified": WOONSTRAAT, "road": WOONSTRAAT,
    # Hoofdweg (doorgaande wegen)
    "motorway": HOOFDWEG, "motorway_link": HOOFDWEG,
    "trunk": HOOFDWEG, "trunk_link": HOOFDWEG,
    "primary": HOOFDWEG, "primary_link": HOOFDWEG,
    "secondary": HOOFDWEG, "secondary_link": HOOFDWEG,
    "tertiary": HOOFDWEG, "tertiary_link": HOOFDWEG,
}

# Een weg telt mee als hij binnen deze afstand (meter) van het punt ligt.
MAX_SNAP_M = 35.0

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Laatste foutreden van de OSM-call, zodat de UI kan tonen WAAROM het faalde.
_LAST_FETCH_ERROR: str | None = None


# --------------------------------------------------------------------------- #
# Hulpfuncties geometrie
# --------------------------------------------------------------------------- #


def _meters_per_degree(lat0: float) -> tuple[float, float]:
    """Benadering meters per graad rond een referentiebreedtegraad."""
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat0))
    return m_lat, m_lon


def _point_segment_dist_m(px_, py_, ax, ay, bx, by, m_lat, m_lon):
    """Afstand (m) van punt P tot lijnsegment A-B in lokale meters.

    Coordinaten in graden worden lokaal naar meters geschaald zodat de
    afstand klopt voor de korte segmenten waar het hier om gaat.
    """
    pxm, pym = px_ * m_lon, py_ * m_lat
    axm, aym = ax * m_lon, ay * m_lat
    bxm, bym = bx * m_lon, by * m_lat

    dx, dy = bxm - axm, bym - aym
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        return math.hypot(pxm - axm, pym - aym)

    t = ((pxm - axm) * dx + (pym - aym) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = axm + t * dx, aym + t * dy
    return math.hypot(pxm - cx, pym - cy)


# --------------------------------------------------------------------------- #
# OSM ophalen
# --------------------------------------------------------------------------- #


def _fetch_osm_ways(min_lat, min_lon, max_lat, max_lon):
    """Haal alle relevante highways binnen de bbox op via Overpass.

    Retourneert een lijst van (categorie, [(lat, lon), ...]) of None bij
    een fout.
    """
    pad = 0.003  # ~300 m marge zodat wegen aan de rand meedoen
    bbox = (min_lat - pad, min_lon - pad, max_lat + pad, max_lon + pad)

    wanted = "|".join(sorted(set(HIGHWAY_MAP.keys())))
    query = (
        "[out:json][timeout:60];"
        f'(way["highway"~"^({wanted})$"]'
        f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););"
        "out geom;"
    )
    import urllib.parse

    data = "data=" + urllib.parse.quote(query)

    global _LAST_FETCH_ERROR
    _LAST_FETCH_ERROR = None
    errors: list[str] = []
    payload = None

    for endpoint in OVERPASS_ENDPOINTS:
        host = endpoint.split("/")[2]
        try:
            req = urllib.request.Request(
                endpoint,
                data=data.encode("utf-8"),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    # Overpass-mirrors weigeren regelmatig requests zonder UA.
                    "User-Agent": "DFC-visualisatie/1.0 (HvA studieproject)",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001 - reden bewust tonen in UI
            errors.append(f"{host}: {exc!r}")
            payload = None
            time.sleep(1)
            continue
    else:
        _LAST_FETCH_ERROR = " | ".join(errors) or "alle endpoints faalden"
        return None

    if payload is None:
        _LAST_FETCH_ERROR = " | ".join(errors) or "geen antwoord van Overpass"
        return None

    ways = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        hw = el.get("tags", {}).get("highway")
        cat = HIGHWAY_MAP.get(hw)
        if cat is None:
            continue
        geom = el.get("geometry") or []
        pts = [(g["lat"], g["lon"]) for g in geom if "lat" in g]
        if len(pts) >= 2:
            ways.append((cat, pts))

    if not ways:
        # Call gelukt, maar nul bruikbare wegen: ander probleem dan een
        # netwerkfout (bv. lege/foute bbox of query).
        n = len(payload.get("elements", []))
        _LAST_FETCH_ERROR = (
            f"call gelukt maar 0 bruikbare wegen gevonden "
            f"(Overpass gaf {n} elementen terug)"
        )
        return None

    return ways


# --------------------------------------------------------------------------- #
# Classificatie
# --------------------------------------------------------------------------- #


def _classify_against_ways(lat, lon, ways, m_lat, m_lon):
    """Geef (categorie, afstand_m) van de dichtstbijzijnde weg, of (None, inf)."""
    best_cat, best_d = None, float("inf")
    for cat, pts in ways:
        for i in range(len(pts) - 1):
            ay, ax = pts[i]
            by, bx = pts[i + 1]
            d = _point_segment_dist_m(lon, lat, ax, ay, bx, by, m_lat, m_lon)
            if d < best_d:
                best_d, best_cat = d, cat
    return best_cat, best_d


def _nearest_known_fallback(lats, lons, classes):
    """Vul ontbrekende klasses met die van het dichtstbijzijnde bekende punt."""
    classes = list(classes)
    known_idx = [i for i, c in enumerate(classes) if c in CATEGORIES]
    if not known_idx:
        return classes  # niets bekend -> niets te doen

    klat = np.array([lats[i] for i in known_idx])
    klon = np.array([lons[i] for i in known_idx])
    for i, c in enumerate(classes):
        if c in CATEGORIES:
            continue
        d2 = (klat - lats[i]) ** 2 + (klon - lons[i]) ** 2
        classes[i] = classes[known_idx[int(np.argmin(d2))]]
    return classes


def _geometric_fallback(df: pd.DataFrame) -> pd.Series:
    """Noodoplossing als OSM onbereikbaar is.

    Zonder wegdata kan het type niet bepaald worden. We kennen alle punten
    daarom "Onbekend" toe (i.p.v. een misleidend "Hoofdweg") zodat de kaart
    eerlijk laat zien dat de classificatie ontbreekt. De app blijft werken.
    """
    return pd.Series([ONBEKEND] * len(df), index=df.index)


def enrich_with_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg een kolom ``classificatie`` toe met Voetpad/Fietspad/Hoofdweg.

    Punten zonder bruikbare weg binnen MAX_SNAP_M krijgen de classificatie
    van het dichtstbijzijnde punt dat wel geclassificeerd is.
    """
    out = df.copy()
    if out.empty or "latitude" not in out or "longitude" not in out:
        out["classificatie"] = pd.Series(dtype=object)
        return out

    lats = out["latitude"].to_numpy(dtype=float)
    lons = out["longitude"].to_numpy(dtype=float)
    valid = ~(np.isnan(lats) | np.isnan(lons))
    if not valid.any():
        out["classificatie"] = None
        return out

    lat0 = float(np.nanmean(lats[valid]))
    m_lat, m_lon = _meters_per_degree(lat0)

    ways = _fetch_osm_ways(
        float(np.nanmin(lats[valid])),
        float(np.nanmin(lons[valid])),
        float(np.nanmax(lats[valid])),
        float(np.nanmax(lons[valid])),
    )

    if ways is None:
        if st is not None:
            st.warning(
                "OSM (Overpass) niet bereikbaar - classificatie gebruikt "
                "een eenvoudige terugvaloptie. Probeer later opnieuw of "
                "controleer de netwerkinstellingen."
            )
        out["classificatie"] = _geometric_fallback(out).values
        out.attrs["osm_ok"] = False  # niet cachen: bij herstart opnieuw proberen
        return out

    raw = []
    for i in range(len(out)):
        if not valid[i]:
            raw.append(None)
            continue
        cat, dist = _classify_against_ways(
            lats[i], lons[i], ways, m_lat, m_lon
        )
        raw.append(cat if (cat is not None and dist <= MAX_SNAP_M) else None)

    filled = _nearest_known_fallback(lats, lons, raw)
    out["classificatie"] = filled
    out.attrs["osm_ok"] = True  # geldig OSM-resultaat: mag gecached worden
    return out


# --------------------------------------------------------------------------- #
# Samenvattingsgrafiek
# --------------------------------------------------------------------------- #


def zone_summary_chart(df: pd.DataFrame, metric_col: str, metric_label: str):
    """Gemiddelde meetwaarde per wegtype als staafdiagram (of None)."""
    if px is None or df.empty or "classificatie" not in df:
        return None
    if metric_col not in df:
        return None

    grp = (
        df.dropna(subset=["classificatie", metric_col])
        .groupby("classificatie")[metric_col]
        .mean()
        .reindex(CATEGORIES)
        .dropna()
        .reset_index()
    )
    if grp.empty:
        return None

    fig = px.bar(
        grp,
        x="classificatie",
        y=metric_col,
        labels={"classificatie": "Wegtype", metric_col: metric_label},
        color="classificatie",
        color_discrete_map={
            VOETPAD: "#2ca02c",
            FIETSPAD: "#1f77b4",
            WOONSTRAAT: "#ff7f0e",
            HOOFDWEG: "#d62728",
        },
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        showlegend=False,
    )
    return fig
