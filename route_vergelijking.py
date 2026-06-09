"""
route_vergelijking.py
---------------------
Vergelijkt de drie meetlopen op basis van afstand langs de route.
Toont temperatuur en luchtvochtigheid per meting, met annotaties
voor bekende plekken (Artis, Nieuwmarkt).
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Bekende locaties langs de route (bij benadering in meters vanaf start)
# Mauritskade start ~0m, langs Artis ~800m, Nieuwmarkt ~2000m
ROUTE_ANNOTATIES = [
    {"afstand_m": 0,    "label": "Mauritskade (start)"},
    {"afstand_m": 800,  "label": "Langs Artis"},
    {"afstand_m": 2000, "label": "Nieuwmarkt"},
]

METING_KLEUREN = {
    1: "#e8590c",   # oranje
    2: "#2ca02c",   # groen
    3: "#1f77b4",   # blauw
}

METING_LABELS = {
    1: "Meting 1 (18 mei, regen)",
    2: "Meting 2 (27 mei)",
    3: "Meting 3 (28 mei, warm)",
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Afstand in meters tussen twee GPS-punten."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def voeg_afstand_toe(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg cumulatieve afstand (meters) toe per meting."""
    resultaat = []
    for meting_nr, sub in df.groupby("meting"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        dists = [0.0]
        for i in range(1, len(sub)):
            d = haversine(
                sub.loc[i - 1, "latitude"], sub.loc[i - 1, "longitude"],
                sub.loc[i, "latitude"],     sub.loc[i, "longitude"],
            )
            dists.append(dists[-1] + d)
        sub["afstand_m"] = dists
        resultaat.append(sub)
    return pd.concat(resultaat).reset_index(drop=True)


def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    """Voortschrijdend gemiddelde om GPS-ruis te dempen."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def toon_route_vergelijking(df: pd.DataFrame):
    """Hoofdfunctie — toont de vergelijkingsgrafiek in Streamlit."""

    st.subheader("Vergelijking drie meetlopen op basis van route-afstand")
    st.caption(
        "X-as = afstand in meters vanaf de Mauritskade. "
        "Alle metingen lopen dezelfde route (~2 km), "
        "zodat omgevingseffecten direct vergelijkbaar zijn."
    )

    df = voeg_afstand_toe(df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Temperatuur (°C)", "Luchtvochtigheid (%)"),
        vertical_spacing=0.10,
    )

    for meting_nr in sorted(df["meting"].unique()):
        sub = df[df["meting"] == meting_nr].sort_values("afstand_m")
        kleur = METING_KLEUREN.get(meting_nr, "#999")
        label = METING_LABELS.get(meting_nr, f"Meting {meting_nr}")

        # Temperatuur
        fig.add_trace(
            go.Scatter(
                x=sub["afstand_m"],
                y=smooth(sub["tempC"]),
                mode="lines",
                name=label,
                line=dict(color=kleur, width=2),
                legendgroup=f"m{meting_nr}",
                hovertemplate="Afstand: %{x:.0f}m<br>Temp: %{y:.1f}°C<extra>" + label + "</extra>",
            ),
            row=1, col=1,
        )

        # Luchtvochtigheid
        fig.add_trace(
            go.Scatter(
                x=sub["afstand_m"],
                y=smooth(sub["humidity"]),
                mode="lines",
                name=label,
                line=dict(color=kleur, width=2, dash="dot"),
                legendgroup=f"m{meting_nr}",
                showlegend=False,
                hovertemplate="Afstand: %{x:.0f}m<br>Vocht: %{y:.1f}%<extra>" + label + "</extra>",
            ),
            row=2, col=1,
        )

    # Verticale annotatielijnen voor bekende locaties
    max_afstand = df["afstand_m"].max()
    for ann in ROUTE_ANNOTATIES:
        if ann["afstand_m"] > max_afstand:
            continue
        for row in [1, 2]:
            fig.add_vline(
                x=ann["afstand_m"],
                line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
                row=row, col=1,
            )
        # Label alleen in bovenste subplot
        fig.add_annotation(
            x=ann["afstand_m"],
            y=1.02,
            yref="paper",
            text=ann["label"],
            showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,0.6)"),
            textangle=-30,
            xanchor="left",
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(title_text="Afstand vanaf start (m)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Tabel met gemiddelden per zone
    st.markdown("**Gemiddelden per zone**")
    zones = [
        ("Mauritskade (0–400m)",  0,    400),
        ("Langs Artis (400–1200m)", 400, 1200),
        ("Richting Nieuwmarkt (1200m+)", 1200, 9999),
    ]
    rijen = []
    for zone_label, z_min, z_max in zones:
        rij = {"Zone": zone_label}
        for meting_nr in sorted(df["meting"].unique()):
            sub = df[(df["meting"] == meting_nr) &
                     (df["afstand_m"] >= z_min) &
                     (df["afstand_m"] < z_max)]
            if not sub.empty:
                rij[f"Temp M{meting_nr} (°C)"]  = f"{sub['tempC'].mean():.1f}"
                rij[f"Vocht M{meting_nr} (%)"]  = f"{sub['humidity'].mean():.1f}"
            else:
                rij[f"Temp M{meting_nr} (°C)"]  = "—"
                rij[f"Vocht M{meting_nr} (%)"]  = "—"
        rijen.append(rij)

    st.dataframe(pd.DataFrame(rijen), use_container_width=True, hide_index=True)

    st.caption(
        "💧 Meting 1 (18 mei) begon met regen — de hogere luchtvochtigheid "
        "en lagere temperatuur zijn zichtbaar in de grafiek en tabel. "
        "Vergelijk Meting 2 en 3 voor de droge omstandigheden."
    )


def toon_correlatie_grafiek(df: pd.DataFrame):
    """Scatterplot temperatuur vs luchtvochtigheid per meting."""

    st.subheader("Correlatie temperatuur & luchtvochtigheid")
    st.caption(
        "Elk punt is één meting (5-secondeninterval). "
        "De negatieve helling toont dat hogere temperatuur "
        "samengaat met lagere luchtvochtigheid."
    )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    fig = go.Figure()

    for meting_nr in sorted(df["meting"].unique()):
        sub = df[df["meting"] == meting_nr].dropna(subset=["tempC", "humidity"])
        kleur = METING_KLEUREN.get(meting_nr, "#999")
        label = METING_LABELS.get(meting_nr, f"Meting {meting_nr}")

        # Scatterplot punten
        fig.add_trace(
            go.Scatter(
                x=sub["tempC"],
                y=sub["humidity"],
                mode="markers",
                name=label,
                marker=dict(color=kleur, size=4, opacity=0.5),
                legendgroup=f"m{meting_nr}",
                hovertemplate="Temp: %{x:.1f}°C<br>Vocht: %{y:.1f}%<extra>" + label + "</extra>",
            )
        )

        # Trendlijn via lineaire regressie
        x = sub["tempC"].values
        y = sub["humidity"].values
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coef, x_line)

        # Pearson r
        r = np.corrcoef(x, y)[0, 1]

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{label} (r={r:.2f})",
                line=dict(color=kleur, width=2),
                legendgroup=f"m{meting_nr}",
                showlegend=True,
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Temperatuur (°C)",
        yaxis_title="Luchtvochtigheid (%)",
        hovermode="closest",
        legend=dict(orientation="h", y=-0.18),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Correlatietabel
    rijen = []
    for meting_nr in sorted(df["meting"].unique()):
        sub = df[df["meting"] == meting_nr].dropna(subset=["tempC", "humidity"])
        r = np.corrcoef(sub["tempC"].values, sub["humidity"].values)[0, 1]
        rijen.append({
            "Meting": METING_LABELS.get(meting_nr, f"Meting {meting_nr}"),
            "Pearson r": f"{r:.3f}",
            "Interpretatie": (
                "Sterke negatieve correlatie" if r < -0.7 else
                "Matige negatieve correlatie" if r < -0.4 else
                "Zwakke correlatie"
            ),
        })

    st.dataframe(pd.DataFrame(rijen), use_container_width=True, hide_index=True)

    st.caption(
        "Meting 1 (regen) heeft naar verwachting een andere helling dan "
        "Meting 2 en 3 — regen voegt vocht toe onafhankelijk van temperatuur, "
        "wat de correlatie verstoort."
    )