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
    st.write("kolommen na rename:", df.columns.tolist())

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
        lines = response.text.splitlines()

        # Zoek de headerregel (begint met "# STN" of "STN")
        header_idx = None
        for i, line in enumerate(lines):
            stripped = line.lstrip("#").strip()
            if stripped.upper().startswith("STN"):
                header_idx = i
                break

        if header_idx is None:
            # Geen header — gebruik vaste kolomnamen op basis van kolomvolgorde
            data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
            df = pd.read_csv(
                io.StringIO("\n".join(data_lines)),
                header=None,
                names=["STN", "datum", "uur", "T_raw"],
                skipinitialspace=True,
            )
        else:
            # Header gevonden — lees met die header
            header_line = lines[header_idx].lstrip("#").strip()
            data_lines = [
                l for l in lines[header_idx + 1:]
                if not l.startswith("#") and l.strip()
            ]
            df = pd.read_csv(
                io.StringIO(header_line + "\n" + "\n".join(data_lines)),
                skipinitialspace=True,
            )
            df.columns = df.columns.str.strip().str.lstrip("#").str.strip()
            for col in list(df.columns):
                c = col.strip().upper()
                if "YYYYMMDD" in c:
                    df = df.rename(columns={col: "datum"})
                elif c == "HH":
                    df = df.rename(columns={col: "uur"})
                elif c == "T":
                    df = df.rename(columns={col: "T_raw"})

        df["datum"] = df["datum"].astype(str).str.strip()
        df["uur"]   = pd.to_numeric(df["uur"], errors="coerce")
        df["T_raw"] = pd.to_numeric(df["T_raw"], errors="coerce")

        dag_offset  = (df["uur"] == 24).astype(int)
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
            mode="lines",
            name="KNMI referentie (Schiphol)",
            line=dict(color="#1f77b4", dash="dash", width=1.5),
        )
    )
    return fig