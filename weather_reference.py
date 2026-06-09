"""
weather_reference.py
--------------------
Haalt uurtemperaturen op via Open-Meteo (historische data) voor de
dagen van de meetlopen en voegt een referentielijn toe aan een
Plotly-grafiek.

API:  https://archive-api.open-meteo.com/v1/archive
      Geen API-key nodig, gratis publieke dienst.
Locatie: Amsterdam centrum (Mauritskade/Nieuwmarkt)
"""

import requests
import pandas as pd
import streamlit as st

# Coördinaten Amsterdam centrum
LAT = 52.364
LON = 4.910

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


@st.cache_data(show_spinner="Open-Meteo referentiedata ophalen...")
def load_reference_temp(dates: tuple | None = None) -> pd.DataFrame | None:
    """Haal uurtemperaturen op van Open-Meteo voor de opgegeven datums.

    Parameters
    ----------
    dates : tuple van datumstrings (YYYYMMDD), of None om alle meetdagen te laden.

    Returns
    -------
    DataFrame met kolommen 'timestamp' en 'knmi_temp_C', of None bij fout.
    """
    if dates is None:
        dates = ("20260518", "20260527", "20260528")

    # Open-Meteo verwacht YYYY-MM-DD formaat
    def fmt(d):
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    start_date = fmt(min(dates))
    end_date   = fmt(max(dates))

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude":        LAT,
                "longitude":       LON,
                "start_date":      start_date,
                "end_date":        end_date,
                "hourly":          "temperature_2m",
                "timezone":        "Europe/Amsterdam",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.warning(f"Open-Meteo data kon niet worden opgehaald: {e}")
        return None

    try:
        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp":    pd.to_datetime(hourly["time"]),
            "knmi_temp_C":  hourly["temperature_2m"],
        })

        # Filter op alleen de gevraagde datums
        df["datum"] = df["timestamp"].dt.strftime("%Y%m%d")
        df = df[df["datum"].isin(dates)].drop(columns="datum")

        return df.dropna().reset_index(drop=True)

    except Exception as e:
        st.warning(f"Open-Meteo data kon niet worden verwerkt: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def add_reference_to_chart(fig, dff: pd.DataFrame, reference_temp: pd.DataFrame):
    """Voeg Open-Meteo referentielijn toe aan een Plotly-figuur."""
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
            mode="lines+markers",
            marker=dict(size=6),
            name="Open-Meteo referentie (Amsterdam)",
            line=dict(color="#1f77b4", dash="dash", width=1.5),
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} °C<extra>Open-Meteo Amsterdam</extra>",
        )
    )
    fig.update_layout(hovermode="x unified")
    return fig


def warmte_eiland_analyse(dff: pd.DataFrame, reference_temp: pd.DataFrame):
    """Statische analyse van het stedelijk warmte-eiland effect."""
    if reference_temp is None or reference_temp.empty:
        return

    # Resample meting naar uurgemiddelden
    meting_uur = (
        dff.set_index("timestamp")["tempC"]
        .resample("1h")
        .mean()
        .reset_index()
        .rename(columns={"tempC": "meting_C"})
    )

    ref = reference_temp.copy()
    ref["timestamp"]      = ref["timestamp"].dt.floor("h")
    meting_uur["timestamp"] = meting_uur["timestamp"].dt.floor("h")

    merged = meting_uur.merge(ref, on="timestamp", how="inner")
    if merged.empty:
        st.info("Geen overlappende uurdata tussen meting en referentie.")
        return

    merged["verschil"] = merged["meting_C"] - merged["knmi_temp_C"]

    gem_verschil = merged["verschil"].mean()
    max_verschil = merged["verschil"].max()
    max_tijdstip = merged.loc[merged["verschil"].idxmax(), "timestamp"]
    max_meting   = merged.loc[merged["verschil"].idxmax(), "meting_C"]
    max_knmi     = merged.loc[merged["verschil"].idxmax(), "knmi_temp_C"]

    st.subheader("Stedelijk warmte-eiland analyse")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Gemiddeld verschil",
        f"+{gem_verschil:.1f} °C",
        help="Gemiddelde van (jouw meting − Open-Meteo referentie) per uur"
    )
    c2.metric(
        "Maximaal verschil",
        f"+{max_verschil:.1f} °C",
        help=f"Hoogste verschil gemeten om {max_tijdstip.strftime('%H:%M')}"
    )
    c3.metric(
        "Tijdstip maximum",
        max_tijdstip.strftime("%H:%M"),
        help=f"Meting: {max_meting:.1f} °C  |  Referentie: {max_knmi:.1f} °C"
    )

    st.markdown("**Waarom is het centrum warmer dan de buitenrand?**")
    st.markdown("""
De route loopt van de **Mauritskade** langs **Artis** naar de **Nieuwmarkt** —
een gradiënt van relatief open naar dicht bebouwd, die zichtbaar is in de
temperatuurmeting:

- **Mauritskade (startpunt)** — open ligging langs het water en brede kade
  zorgen voor enige koeling door verdamping en wind. Dit is het koelste
  deel van de route.
- **Langs Artis** — het parkgroen en de bomen langs Artis dempen de
  opwarming tijdelijk via verdamping (evapotranspiratie) en schaduw.
- **Nieuwmarkt (eindpunt)** — dichte bebouwing, weinig groen, veel toeristen
  en horeca. Steen en asfalt slaan overdag warmte op. Wind wordt
  geblokkeerd door smalle straatjes.
- **Verharding** — het aandeel verharding neemt toe richting de Nieuwmarkt,
  met minder verdampingskoeling als gevolg.
- **Menselijke warmtebronnen** — verkeer, airconditioning en mensen leveren
  extra warmte, geconcentreerd in het drukke centrum.
- **Open-Meteo referentie** — gebaseerd op een gridpunt boven Amsterdam
  centrum (~1km² gemiddelde), wat de stedelijke opwarming deels al
  meeneemt maar minder gevoelig is voor lokale straatniveau-effecten.
""")

    st.caption(
        f"Gebaseerd op {len(merged)} uurgemiddelden · "
        f"Meting: {merged['meting_C'].mean():.1f} °C gemiddeld · "
        f"Open-Meteo: {merged['knmi_temp_C'].mean():.1f} °C gemiddeld"
    )