"""
weather_reference.py
--------------------
Haalt uurtemperaturen op van het KNMI voor de dagen van de meetlopen
en voegt een referentielijn toe aan een Plotly-grafiek.

Station: Schiphol (240) — dichtstbijzijnde KNMI-station bij Purmerend.
API:     https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
         Geen API-key nodig, gratis publieke dienst.
"""

import io
import requests
import pandas as pd
import streamlit as st


# KNMI station dichtstbijzijnde de meetlocatie (Purmerend)
KNMI_STATION = "240"  # Schiphol
KNMI_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"


@st.cache_data(show_spinner="KNMI referentiedata ophalen...")
def load_reference_temp(dates: tuple[str, ...] | None = None) -> pd.DataFrame | None:
    """Haal uurtemperaturen op van KNMI voor de opgegeven datums.

    Parameters
    ----------
    dates : tuple van datumstrings (YYYYMMDD), of None om alle meetdagen te laden.

    Returns
    -------
    DataFrame met kolommen 'timestamp' (tz-naive) en 'knmi_temp_C',
    of None als de aanvraag mislukt.
    """
    if dates is None:
        # Standaard: alle drie de meetdagen
        dates = ("20260518", "20260527", "20260528")

    # Bouw start/end: begin van eerste dag t/m einde van laatste dag
    start = min(dates) + "01"
    end   = max(dates) + "24"

    try:
        response = requests.post(
            KNMI_URL,
            data={
                "start": start,
                "end":   end,
                "stns":  KNMI_STATION,
                "vars":  "T",       # T = temperatuur in 0.1 °C
                "fmt":   "csv",
            },
            timeout=20,
        )
        response.raise_for_status()
    except Exception as e:
        st.warning(f"KNMI data kon niet worden opgehaald: {e}")
        return None

    # Sla commentaarregels over (beginnen met #)
    lines = [l for l in response.text.splitlines() if not l.startswith("#")]
    if not lines:
        return None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            skipinitialspace=True,
        )
        st.write("KNMI kolommen:", df.columns.tolist())
        st.write(df.head(3))
        # Kolomnamen opschonen (KNMI geeft soms spaties mee)
        df.columns = df.columns.str.strip()

        # Verwachte kolommen: STN, YYYYMMDD, HH, T
        df = df.rename(columns={"YYYYMMDD": "datum", "HH": "uur", "T": "T_raw"})
        df["datum"] = df["datum"].astype(str)
        df["uur"]   = pd.to_numeric(df["uur"], errors="coerce")
        df["T_raw"] = pd.to_numeric(df["T_raw"], errors="coerce")

        # KNMI uur 24 = middernacht volgende dag → zet om naar 0
        df["uur_adj"] = df["uur"].mod(24)
        dag_offset = (df["uur"] == 24).astype(int)

        df["timestamp"] = pd.to_datetime(df["datum"], format="%Y%m%d") \
            + pd.to_timedelta(df["uur_adj"], unit="h") \
            + pd.to_timedelta(dag_offset, unit="D")

        df["knmi_temp_C"] = df["T_raw"] / 10.0  # 0.1°C → °C

        # Filter op alleen de gevraagde datums
        df = df[df["datum"].isin(dates)].copy()

        return df[["timestamp", "knmi_temp_C"]].dropna().reset_index(drop=True)

    except Exception as e:
        st.warning(f"KNMI data kon niet worden verwerkt: {e}")
        return None


def add_reference_to_chart(fig, dff: pd.DataFrame, reference_temp: pd.DataFrame):
    """Voeg KNMI referentielijn toe aan een Plotly-figuur.

    Koppelt op basis van timestamp (nearest-hour match) zodat de
    referentielijn alleen de tijdspanne van de meting beslaat.
    """
    if reference_temp is None or reference_temp.empty:
        return fig

    # Bepaal het tijdsbereik van de huidige meting
    t_start = dff["timestamp"].min().floor("h")
    t_end   = dff["timestamp"].max().ceil("h")

    ref = reference_temp[
        (reference_temp["timestamp"] >= t_start) &
        (reference_temp["timestamp"] <= t_end)
    ].copy()

    if ref.empty:
        return fig

    import plotly.graph_objects as go
    fig.add_trace(
        go.Scatter(
            x=ref["timestamp"],
            y=ref["knmi_temp_C"],
            mode="lines",
            name="KNMI referentie (Schiphol)",
            line=dict(color="#1f77b4", dash="dash", width=1.5),
        )
    )
    return fig