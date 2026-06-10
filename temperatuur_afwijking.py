"""
Lucht 2 — Afwijking van de gemiddelde temperatuur per dag
----------------------------------------------------------
Doel: Visualiseert per meetdag hoe de gemeten temperatuur afwijkt van
      het daggemiddelde, in de huisstijl van de DFC-eindpresentatie.
In:   DATA.CSV (timestamp, gps_time, lat, ns, lon, ew, co2_ppm, tempC, humidity)
Uit:  temperatuur_en_boxplot.png — combifiguur: lijnplot + boxplot
      temperatuur_met_gemiddelde.png — temperatuur per dag met gemiddelde lijn
      afwijking_per_dag.png  — verloop van de afwijking per dag (lijnplot)
      afwijking_boxplot.png  — spreiding van de afwijking per dag (boxplot)

Gebruik: python temperatuur_afwijking.py [pad/naar/DATA.CSV]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### Thema — kleuren uit de Lucht 2 / DFC-presentatie
NAVY = "#14304D"        # donkerblauw (titels, kaarten)
TEAL = "#16A5B8"        # accentkleur (header-balk)
BLUE = "#1F6FB2"        # middenblauw
GREEN = "#1B9A6C"       # groen (derde kaartkleur)
ORANGE = "#F5A623"      # oranje (backup/attentie)
BG = "#F0F4F8"          # lichte achtergrond van de slides
GRID = "#D7E0EA"

DAG_KLEUREN = [NAVY, TEAL, BLUE, GREEN, ORANGE]


def laad_data(pad: str) -> pd.DataFrame:
    ### Inlezen en opschonen
    df = pd.read_csv(pad)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "tempC"]).copy()

    df["datum"] = df["timestamp"].dt.date

    # Klok-glitch van de sensor: metingen met een datum v\u00f3\u00f3r 2020
    # (bijv. 21-04-2015, niet-gesynchroniseerde GPS-klok) verwijderen
    df = df[df["timestamp"].dt.year >= 2020].copy()

    # Dagen met te weinig punten overslaan
    telling = df["datum"].value_counts()
    geldige_dagen = telling[telling >= 30].index
    df = df[df["datum"].isin(geldige_dagen)].copy()

    ### Afwijking t.o.v. het daggemiddelde
    df["dag_gem"] = df.groupby("datum")["tempC"].transform("mean")
    df["afwijking"] = df["tempC"] - df["dag_gem"]

    # Tijd-as: minuten sinds de start van de meting, zodat de dagen
    # over elkaar geplot kunnen worden ondanks verschillende starttijden
    df["minuten"] = (
        df["timestamp"] - df.groupby("datum")["timestamp"].transform("min")
    ).dt.total_seconds() / 60
    return df


def stijl_toepassen() -> None:
    ### Seaborn/matplotlib-thema in presentatiestijl
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "grid.color": GRID,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "font.family": "sans-serif",
        "legend.frameon": False,
    })


def maak_temperatuurplot(df: pd.DataFrame, uitpad: Path) -> None:
    ### Gemeten temperatuur per dag met het daggemiddelde als lijn erdoorheen
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for kleur, (datum, groep) in zip(
        DAG_KLEUREN, df.groupby("datum", sort=True)
    ):
        groep = groep.sort_values("minuten")
        gem = groep["dag_gem"].iloc[0]
        label = f"{datum.strftime('%d %b %Y')} (gem. {gem:.1f}\u00b0C)"
        ax.plot(groep["minuten"], groep["tempC"], color=kleur,
                linewidth=2, label=label)
        # Gemiddelde lijn over de duur van de eigen meting
        ax.hlines(gem, groep["minuten"].min(), groep["minuten"].max(),
                  color=kleur, linewidth=1.4, linestyle="--", alpha=0.7)
        ax.annotate(f"{gem:.1f}\u00b0C", color=kleur, fontsize=9,
                    fontweight="bold",
                    xy=(groep["minuten"].max(), gem),
                    xytext=(5, 0), textcoords="offset points", va="center")

    ax.set_title("Temperatuur per meetdag met daggemiddelde",
                 pad=14, loc="left")
    ax.set_xlabel("Minuten sinds start van de meting")
    ax.set_ylabel("Temperatuur (\u00b0C)")
    ax.legend(title="Meetdag", loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)

    fig.subplots_adjust(top=0.86)
    fig.patches.append(plt.Rectangle((0, 0.97), 1, 0.03, color=TEAL,
                                     transform=fig.transFigure, zorder=5))
    fig.text(0.99, 0.01, "Lucht 2 \u00b7 DFC", ha="right",
             fontsize=8, color=NAVY, alpha=0.6)

    fig.savefig(uitpad, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def maak_lijnplot(df: pd.DataFrame, uitpad: Path) -> None:
    ### Verloop van de afwijking per dag over de tijd van de dag
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for kleur, (datum, groep) in zip(
        DAG_KLEUREN, df.groupby("datum", sort=True)
    ):
        groep = groep.sort_values("minuten")
        start = groep["timestamp"].min().strftime("%H:%M")
        label = (f"{datum.strftime('%d %b %Y')} \u00b7 start {start} "
                 f"(gem. {groep['dag_gem'].iloc[0]:.1f}\u00b0C)")
        ax.plot(groep["minuten"], groep["afwijking"], color=kleur,
                linewidth=2, label=label)

    ax.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)

    ax.set_title("Afwijking van de gemiddelde temperatuur per dag",
                 pad=14, loc="left")
    ax.set_xlabel("Minuten sinds start van de meting")
    ax.set_ylabel("Afwijking t.o.v. daggemiddelde (\u00b0C)")
    ax.legend(title="Meetdag", loc="best")

    # Teal accentbalk bovenaan, zoals op de slides
    fig.subplots_adjust(top=0.86)
    fig.patches.append(plt.Rectangle((0, 0.97), 1, 0.03, color=TEAL,
                                     transform=fig.transFigure, zorder=5))
    fig.text(0.99, 0.01, "Lucht 2 \u00b7 DFC", ha="right",
             fontsize=8, color=NAVY, alpha=0.6)

    fig.savefig(uitpad, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def maak_boxplot(df: pd.DataFrame, uitpad: Path) -> None:
    ### Spreiding van de afwijking per dag
    fig, ax = plt.subplots(figsize=(9, 5.5))

    volgorde = sorted(df["datum"].unique())
    labels = [d.strftime("%d %b %Y") for d in volgorde]
    palette = {d: k for d, k in zip(volgorde, DAG_KLEUREN)}

    sns.boxplot(data=df, x="datum", y="afwijking", order=volgorde,
                hue="datum", palette=palette, legend=False,
                width=0.5, fliersize=2, ax=ax)

    ax.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(volgorde)))
    ax.set_xticklabels(labels)

    ax.set_title("Spreiding van de temperatuurafwijking per meetdag",
                 pad=14, loc="left")
    ax.set_xlabel("Meetdag")
    ax.set_ylabel("Afwijking t.o.v. daggemiddelde (\u00b0C)")

    fig.subplots_adjust(top=0.86)
    fig.patches.append(plt.Rectangle((0, 0.97), 1, 0.03, color=TEAL,
                                     transform=fig.transFigure, zorder=5))
    fig.text(0.99, 0.01, "Lucht 2 \u00b7 DFC", ha="right",
             fontsize=8, color=NAVY, alpha=0.6)

    fig.savefig(uitpad, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)



def maak_combifiguur(df: pd.DataFrame, uitpad: Path) -> None:
    ### Eén figuur: temperatuurverloop met daggemiddelde + boxplot van de afwijking
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [3, 2]}
    )

    volgorde = sorted(df["datum"].unique())
    palette = {d: k for d, k in zip(volgorde, DAG_KLEUREN)}

    ## Links: temperatuur per dag met gemiddelde lijn erdoorheen
    for datum in volgorde:
        groep = df[df["datum"] == datum].sort_values("minuten")
        kleur = palette[datum]
        gem = groep["dag_gem"].iloc[0]
        ax1.plot(groep["minuten"], groep["tempC"], color=kleur,
                 linewidth=2, label=f"{datum.strftime('%d %b %Y')} (gem. {gem:.1f}\u00b0C)")
        ax1.hlines(gem, groep["minuten"].min(), groep["minuten"].max(),
                   color=kleur, linewidth=1.4, linestyle="--", alpha=0.7)
        ax1.annotate(f"{gem:.1f}\u00b0C", color=kleur, fontsize=9,
                     fontweight="bold",
                     xy=(groep["minuten"].max(), gem),
                     xytext=(5, 0), textcoords="offset points", va="center")

    ax1.set_title("Temperatuur per meetdag met daggemiddelde",
                  pad=12, loc="left")
    ax1.set_xlabel("Minuten sinds start van de meting")
    ax1.set_ylabel("Temperatuur (\u00b0C)")
    ax1.legend(title="Meetdag", loc="upper left", fontsize=9)

    ## Rechts: spreiding van de afwijking per dag
    labels = [d.strftime("%d %b") for d in volgorde]
    sns.boxplot(data=df, x="datum", y="afwijking", order=volgorde,
                hue="datum", palette=palette, legend=False,
                width=0.5, fliersize=2, ax=ax2)
    ax2.set_facecolor("white")
    ax2.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_xticks(range(len(volgorde)))
    ax2.set_xticklabels(labels)
    ax2.set_title("Spreiding van de afwijking per meetdag",
                  pad=12, loc="left")
    ax2.set_xlabel("Meetdag")
    ax2.set_ylabel("Afwijking t.o.v. daggemiddelde (\u00b0C)")

    ## Opmaak in presentatiestijl
    fig.subplots_adjust(top=0.86, wspace=0.25)
    fig.patches.append(plt.Rectangle((0, 0.97), 1, 0.03, color=TEAL,
                                     transform=fig.transFigure, zorder=5))
    fig.text(0.99, 0.01, "Lucht 2 \u00b7 DFC", ha="right",
             fontsize=8, color=NAVY, alpha=0.6)

    fig.savefig(uitpad, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    pad = sys.argv[1] if len(sys.argv) > 1 else "DATA.CSV"
    uitmap = Path(".")

    df = laad_data(pad)
    stijl_toepassen()

    maak_combifiguur(df, uitmap / "temperatuur_en_boxplot.png")
    maak_temperatuurplot(df, uitmap / "temperatuur_met_gemiddelde.png")
    maak_lijnplot(df, uitmap / "afwijking_per_dag.png")
    maak_boxplot(df, uitmap / "afwijking_boxplot.png")

    ### Korte samenvatting in de terminal
    overzicht = (
        df.groupby("datum")["tempC"]
        .agg(gemiddelde="mean", minimum="min", maximum="max", n="count")
        .round(2)
    )
    print("Daggemiddelden temperatuur (\u00b0C):")
    print(overzicht.to_string())
    print("\nGrafieken opgeslagen: temperatuur_en_boxplot.png, temperatuur_met_gemiddelde.png, afwijking_per_dag.png, afwijking_boxplot.png")


if __name__ == "__main__":
    main()
