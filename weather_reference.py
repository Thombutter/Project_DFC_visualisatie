"""
weather_reference.py
--------------------
Referentie-temperatuur uit Open-Meteo.

Databron: https://open-meteo.com/ (geen API-sleutel nodig). We gebruiken
de archief-API (ERA5 reanalyse) voor historische uren, en vallen terug
op de forecast-API voor recente dagen die nog niet in het archief staan.

`add_reference_to_chart` voegt een gestippelde referentielijn toe aan de
bestaande Plotly-figuur met de gemeten temperatuur, zodat je de
sensormeting kunt vergelijken met de officiele buitentemperatuur.
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _parse_date(date) -> _dt.date | None:
    """Accepteer 'YYYYMMDD', 'YYYY-MM-DD', date/datetime of None."""
    if date is None:
        return None
    if isinstance(date, _dt.datetime):
        return date.date()
    if isinstance(date, _dt.date):
        return date
    s = str(date).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _http_json(url: str, params: dict):
    """Eenvoudige GET die JSON teruggeeft, of None bij een fout."""
    query = urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(f"{url}?{query}", timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def load_reference_temp(date=None, lat: float | None = None,
                        lon: float | None = None) -> pd.DataFrame | None:
    """Haal uurlijkse referentie-temperatuur op via Open-Meteo.

    Parameters
    ----------
    date : str | date | None
        Dag waarvoor de referentie wordt opgehaald. Mag ook het
        'YYYYMMDD'-formaat zijn. None -> dan wordt later (in
        add_reference_to_chart) de datum uit de meetdata afgeleid.
    lat, lon : float | None
        Locatie. None -> standaard Beverwijk/IJmuiden-omgeving, wordt
        zo nodig in add_reference_to_chart overschreven met de
        routelocatie uit de meetdata.

    Returns
    -------
    DataFrame met kolommen ['time', 'temp_ref'] of None.
    """
    day = _parse_date(date)
    if day is None:
        # Geen vaste datum: signaleer dat de datum later uit de data komt.
        return pd.DataFrame(columns=["time", "temp_ref"])

    if lat is None or lon is None:
        lat, lon = 52.46, 4.61  # Beverwijk/IJmuiden-omgeving

    return _fetch_hourly_temp(lat, lon, day)


def _fetch_hourly_temp(lat: float, lon: float,
                        day: _dt.date) -> pd.DataFrame | None:
    """Uurlijkse 2m-temperatuur voor een dag/locatie via Open-Meteo."""
    iso = day.isoformat()
    common = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "temperature_2m",
        "timezone": "Europe/Amsterdam",
        "start_date": iso,
        "end_date": iso,
    }

    # Recente dagen staan nog niet in het ERA5-archief -> forecast-API
    # (die heeft een paar maanden aan verleden via past_days niet, maar
    # wel een 'archive'-achtige dekking voor de afgelopen ~3 maanden).
    age_days = (_dt.date.today() - day).days
    url = ARCHIVE_URL if age_days >= 5 else FORECAST_URL

    payload = _http_json(url, common)
    if payload is None and url != ARCHIVE_URL:
        payload = _http_json(ARCHIVE_URL, common)  # laatste poging
    if not payload or "hourly" not in payload:
        if st is not None:
            st.info(
                "Open-Meteo referentie niet beschikbaar voor deze dag/"
                "locatie - de grafiek toont alleen de sensormeting."
            )
        return None

    times = payload["hourly"].get("time", [])
    temps = payload["hourly"].get("temperature_2m", [])
    if not times or not temps:
        return None

    out = pd.DataFrame(
        {"time": pd.to_datetime(times), "temp_ref": temps}
    ).dropna()
    return out if not out.empty else None


def add_reference_to_chart(fig, dff: pd.DataFrame, ref):
    """Voeg de Open-Meteo referentielijn toe aan de temperatuurgrafiek.

    `ref` is het resultaat van load_reference_temp. Als dat leeg is (geen
    vaste datum opgegeven), wordt de juiste dag en locatie alsnog uit de
    meetdata `dff` afgeleid en hier opgehaald.
    """
    if dff is None or dff.empty or "timestamp" not in dff:
        return fig

    need_fetch = ref is None or (
        isinstance(ref, pd.DataFrame) and ref.empty
    )
    if need_fetch:
        ts = pd.to_datetime(dff["timestamp"], errors="coerce").dropna()
        if ts.empty:
            return fig
        day = ts.iloc[len(ts) // 2].date()  # representatieve dag
        if {"latitude", "longitude"}.issubset(dff.columns):
            lat = float(dff["latitude"].mean())
            lon = float(dff["longitude"].mean())
        else:
            lat, lon = 52.46, 4.61
        ref = _fetch_hourly_temp(lat, lon, day)

    if ref is None or not isinstance(ref, pd.DataFrame) or ref.empty:
        return fig

    # Beperk de referentielijn tot het tijdsbereik van de meting
    ts = pd.to_datetime(dff["timestamp"], errors="coerce")
    tmin, tmax = ts.min(), ts.max()
    sub = ref[(ref["time"] >= tmin - pd.Timedelta(hours=1))
              & (ref["time"] <= tmax + pd.Timedelta(hours=1))]
    if sub.empty:
        sub = ref

    fig.add_scatter(
        x=sub["time"],
        y=sub["temp_ref"],
        mode="lines",
        name="Referentie (Open-Meteo)",
        line=dict(color="#1f77b4", dash="dash", width=2),
    )
    return fig
