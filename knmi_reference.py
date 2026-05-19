"""
knmi_reference.py  –  KNMI uurtemperatuur als referentielijn
-------------------------------------------------------------
Plak dit bestand naast je app.py en voeg de onderstaande
aanroepen toe aan je Streamlit-script.

Hoe werkt het?
  - KNMI biedt een gratis CSV-download via daggegevens.knmi.nl.
  - Station 240 = Schiphol, het dichtstbijzijnde officiële
    KNMI-meetstation voor Amsterdam (~9 km).
  - Temperatuur staat als T in 0.1 °C → we delen door 10.
  - De uurwaarden worden gematcht aan je sensor-timestamps
    via de dichtstbijzijnde volle uur.

Gebruik in je bestaande app
---------------------------
Stap 1: importeer bovenaan app.py:
    from knmi_reference import load_knmi_hourly, add_knmi_to_chart

Stap 2: na je tijdsfilter, laad de KNMI-data:
    knmi = load_knmi_hourly(
        date="20260518",   # aanpassen aan je meetdag  YYYYMMDD
        station=240,       # 240 = Schiphol; 260 = De Bilt
    )

Stap 3: vervang je plotly-grafiek voor temperatuur door:
    if metric_col == "tempC" and knmi is not None:
        fig = add_knmi_to_chart(fig, dff, knmi)

Dat is alles!
"""

import io
import textwrap

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# --------------------------------------------------------------------------- #
# KNMI data ophalen
# --------------------------------------------------------------------------- #

KNMI_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"


@st.cache_data(ttl=3600)
def load_knmi_hourly(date: str, station: int = 240) -> pd.DataFrame | None:
    """
    Haal uurtemperaturen op van KNMI voor één dag.

    Parameters
    ----------
    date    : str  Datum als 'YYYYMMDD', bijv. '20260518'
    station : int  KNMI-stationnummer (240 = Schiphol, 260 = De Bilt)

    Returns
    -------
    DataFrame met kolommen [hour, tempC_knmi] of None bij fout.
    """
    params = {
        "start": date,
        "end": date,
        "stns": station,
        "vars": "T",
        "fmt": "csv",
    }
    try:
        r = requests.get(KNMI_URL, params=params, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        st.warning(f"KNMI-data kon niet worden opgehaald: {exc}")
        return None

    raw = r.text

    # KNMI CSV heeft een commentaarblok bovenaan dat begint met '#'
    lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    if not lines:
        st.warning("KNMI-respons bevat geen data.")
        return None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            skipinitialspace=True,
        )
    except Exception as exc:
        st.warning(f"KNMI CSV kon niet worden geparsed: {exc}")
        return None

    # Kolomnamen zijn inconsistent; normaliseer ze
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"HH", "T"}
    if not required.issubset(df.columns):
        st.warning(
            f"Verwachte kolommen {required} ontbreken in KNMI-data. "
            f"Aanwezig: {list(df.columns)}"
        )
        return None

    df["hour"] = pd.to_numeric(df["HH"], errors="coerce")   # uur 1-24
    df["tempC_knmi"] = pd.to_numeric(df["T"], errors="coerce") / 10.0
    df = df.dropna(subset=["hour", "tempC_knmi"]).reset_index(drop=True)

    # Uur 24 → volgende dag 0:00; voor één-dag-weergave laten we dit staan
    return df[["hour", "tempC_knmi"]]


# --------------------------------------------------------------------------- #
# Koppelen aan sensordata
# --------------------------------------------------------------------------- #

def merge_knmi_with_sensor(
    dff: pd.DataFrame,
    knmi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Voeg KNMI-temperatuur toe aan sensordata op basis van het dichtstbijzijnde uur.

    Parameters
    ----------
    dff  : sensor-DataFrame met kolom 'timestamp'
    knmi : DataFrame van load_knmi_hourly()

    Returns
    -------
    dff met extra kolom 'tempC_knmi'
    """
    dff = dff.copy()
    # Uur van de sensortimestamp (1-24, conform KNMI-notatie)
    dff["_hour"] = dff["timestamp"].dt.hour.replace(0, 24)
    knmi_map = knmi.set_index("hour")["tempC_knmi"].to_dict()
    dff["tempC_knmi"] = dff["_hour"].map(knmi_map)
    dff = dff.drop(columns=["_hour"])
    return dff


# --------------------------------------------------------------------------- #
# Grafiek uitbreiden met KNMI-referentielijn
# --------------------------------------------------------------------------- #

def add_knmi_to_chart(
    fig: go.Figure,
    dff: pd.DataFrame,
    knmi: pd.DataFrame,
) -> go.Figure:
    """
    Voeg een KNMI-referentielijn toe aan een bestaand Plotly-figuur.

    Verwacht dat fig al een trace heeft voor 'tempC' over 'timestamp'.
    """
    merged = merge_knmi_with_sensor(dff, knmi)

    fig.add_trace(
        go.Scatter(
            x=merged["timestamp"],
            y=merged["tempC_knmi"],
            mode="lines",
            name="KNMI Schiphol (referentie)",
            line=dict(color="#1a73e8", width=2, dash="dash"),
        )
    )

    fig.update_layout(legend=dict(orientation="h", y=1.08))
    return fig


# --------------------------------------------------------------------------- #
# Standalone demo (python knmi_reference.py)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "20260518"
    print(f"Ophalen KNMI uurdata voor {date}, station 240 (Schiphol)…")

    params = {"start": date, "end": date,
              "stns": 240, "vars": "T", "fmt": "csv"}
    r = requests.get(KNMI_URL, params=params, timeout=10)
    print("HTTP status:", r.status_code)

    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]
    df["tempC"] = pd.to_numeric(df["T"], errors="coerce") / 10.0
    print(df[["HH", "T", "tempC"]].to_string(index=False))
