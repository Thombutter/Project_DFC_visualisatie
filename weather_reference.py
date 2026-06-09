"""
weather_reference.py
--------------------
Haalt uurtemperaturen op van het KNMI voor de dagen van de meetlopen
en voegt een referentielijn toe aan een Plotly-grafiek.

Station: Schiphol (240) — dichtstbijzijnde KNMI-station bij Amsterdam.
API:     https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
         Geen API-key nodig, gratis publieke dienst.
"""

import io
import re
import requests
import pandas as pd
import streamlit as st

KNMI_STATION = "240"
KNMI_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"


@st.cache_data(show_spinner="KNMI referentiedata ophalen...")
def load_reference_temp(dates: tuple | None = None) -> pd.DataFrame | None:
    if dates is None:
        dates = ("20260518", "20260527", "20260528")

    start = min(dates) + "01"
    end   = max(dates) + "24"

    try:
        response = requests.post(
            KNMI_URL,
            data={
                "start": start,
                "end":   end,
                "stns":  KNMI_STATION,
                "vars":  "T",
                "fmt":   "csv",
            },
            timeout=20,
        )
        response.raise_for_status()
    except Exception as e:
        st.warning(f"KNMI data kon niet worden opgehaald: {e}")
        return None

    try:
        text = response.text

        # Vind de headerregel STN,YYYYMMDD,HH,...
        header_match = re.search(r"STN,YYYYMMDD,HH[^\n]*", text)
        if header_match is None:
            st.warning("KNMI: geen headerregel gevonden.")
            return None

        # Alles na de header is data — maar datarijen staan mogelijk
        # aaneengesloten op één regel, gescheiden door spaties.
        # Splits op patroon: getal,getal (stationsnummer aan het begin)
        data_text = text[header_match.end():]

        # Voeg newline in vóór elk voorkomen van het stationsnummer (240,)
        data_text = re.sub(r"\s+" + KNMI_STATION + r",", "\n" + KNMI_STATION + ",", data_text)
        data_text = data_text.strip()

        header_line = header_match.group(0)
        df = pd.read_csv(
            io.StringIO(header_line + "\n" + data_text),
            skipinitialspace=True,
        )

        # Kolomnamen opschonen
        df.columns = df.columns.str.strip()

        # Hernoem naar interne namen
        df = df.rename(columns={
            "YYYYMMDD": "datum",
            "HH": "uur",
            "T": "T_raw",
        })

        df["datum"] = df["datum"].astype(str).str.strip()
        df["uur"]   = pd.to_numeric(df["uur"], errors="coerce")
        df["T_raw"] = pd.to_numeric(df["T_raw"], errors="coerce")

        dag_offset    = (df["uur"] == 24).astype(int)
        df["uur_adj"] = df["uur"].mod(24)

        df["timestamp"] = (
            pd.to_datetime(df["datum"], format="%Y%m%d")
            + pd.to_timedelta(df["uur_adj"], unit="h")
            + pd.to_timedelta(dag_offset, unit="D")
        )

        df["knmi_temp_C"] = df["T_raw"] / 10.0
        df = df[df["datum"].isin(dates)].copy()

        return df[["timestamp", "knmi_temp_C"]].dropna().reset_index(drop=True)

    except Exception as e:
        st.warning(f"KNMI data kon niet worden verwerkt: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def add_reference_to_chart(fig, dff: pd.DataFrame, reference_temp: pd.DataFrame):
    if reference_temp is None or reference_temp.empty:
        return fig

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
            mode="lines+markers",        # ← markers zodat hover werkt
            marker=dict(size=6),
            name="KNMI referentie (Schiphol)",
            line=dict(color="#1f77b4", dash="dash", width=1.5),
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} °C<extra>KNMI Schiphol</extra>",
        )
    )
    fig.update_layout(hovermode="x unified")  # ← toont beide traces tegelijk
    return fig