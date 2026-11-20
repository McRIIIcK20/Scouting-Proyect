from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml

# Core MCDM models (extracted to core/models.py)
from core.models import (
    ahp_weights as _ahp_weights,
    impute_matrix as _impute_matrix,
    merec as _merec,
    mabac as _mabac,
    topsis as _topsis,
)
from core.methods import (
    RANKING_METHODS,
    WEIGHT_METHODS,
    ahp_consistency,
    anp_weights,
    bwm_weights,
    entropy_weights,
    itara_weights,
    ranks_from_scores,
    swara_weights,
    three_way,
)


# ============================================================
# SCOUTING LAB — DATA-DRIVEN FOOTBALL SCOUTING
# Versión corregida
# ============================================================

st.set_page_config(
    page_title="Scouting Lab | Moneyball",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datos" / "procesado" / "resultados_finales.csv"
HIST_PATH = BASE_DIR / "datos" / "procesado" / "fw_base.csv"
FOTOS_DIR = BASE_DIR / "fotos_accion"
STYLE_PATH = BASE_DIR / "assets" / "style.css"
TOKENS_PATH = BASE_DIR / "assets" / "tokens.css"
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"

with SETTINGS_PATH.open("r", encoding="utf-8") as _settings_file:
    SETTINGS = yaml.safe_load(_settings_file)

AHP_CRITERIA = list(SETTINGS["ahp"]["criteria"])
AHP_BENEFITS = [bool(b) for b in SETTINGS["ahp"]["benefits"]]
AHP_SLIDER_KEYS = [
    "ahp_w_npxg",
    "ahp_w_sot",
    "ahp_w_xa",
    "ahp_w_value",
]


# ============================================================
# ESTILO
# ============================================================

# style.css lleva el sello y la fuente; tokens.css (los colores y la escala
# del tema) se inserta en el marcador para que todo viaje en un solo <style>.
css = STYLE_PATH.read_text(encoding="utf-8").replace(
    "/* @tokens */",
    TOKENS_PATH.read_text(encoding="utf-8"),
)

# st.html oculta los bloques sin altura visible (.empty-html), lo que
# eliminaria el espaciado que dejaba st.markdown; el <div> de 1px con
# margen negativo restaura ese hueco y mantiene el layout identico.
st.html(
    f"<style>{css}</style>"
    '<div aria-hidden="true" style="height:1px;margin-bottom:-1px;"></div>'
)


# Paleta del tema Console en hex (Plotly no lee variables CSS).
# Ámbar = lo que elige una persona (la selección, las gemas);
# azul frío = lo que mide el modelo (series, escalas).
PALETTE = {
    "paper": "#0d1013",
    "paper_2": "#15191c",
    "paper_3": "#06080b",
    "ink": "#eceff2",
    "ink_2": "#c3c8cc",
    "muted": "#9da2a8",
    "rule": "#2d3135",
    "rule_2": "#484e54",
    "accent": "#f7b83d",
    "cool": "#6ebcfc",
    "cool_mid": "#3b79ad",
    "cool_soft": "#142e43",
    "ok": "#61cb7c",
    "bad": "#f66d67",
}
_axis = dict(
    gridcolor=PALETTE["rule"],
    zerolinecolor=PALETTE["rule_2"],
    linecolor=PALETTE["rule"],
    tickfont=dict(color=PALETTE["muted"]),
    title=dict(font=dict(color=PALETTE["ink_2"])),
)
pio.templates["scouting"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Host Grotesk, ui-sans-serif, system-ui, sans-serif",
            color=PALETTE["ink_2"],
            size=13,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            PALETTE["cool"],
            PALETTE["accent"],
            PALETTE["ok"],
            PALETTE["bad"],
            PALETTE["muted"],
        ],
        xaxis=_axis,
        yaxis=_axis,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(gridcolor=PALETTE["rule"], linecolor=PALETTE["rule"], tickfont=dict(color=PALETTE["muted"])),
            angularaxis=dict(gridcolor=PALETTE["rule"], linecolor=PALETTE["rule_2"], tickfont=dict(color=PALETTE["ink_2"])),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["paper_3"],
            bordercolor=PALETTE["rule_2"],
            font=dict(color=PALETTE["ink"], family="Atkinson Hyperlegible Mono, ui-monospace, monospace"),
        ),
        title=dict(font=dict(color=PALETTE["ink"], size=15)),
        legend=dict(font=dict(color=PALETTE["ink_2"])),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=PALETTE["muted"]))),
    )
)
CHART_TEMPLATE = "plotly_dark+scouting"
SCALE_COOL = [PALETTE["cool_soft"], PALETTE["cool_mid"], PALETTE["cool"]]
SCALE_COOL_REV = [PALETTE["cool"], PALETTE["cool_mid"], PALETTE["cool_soft"]]


# ============================================================
# UTILIDADES
# ============================================================

ALIASES = {
    "name": ["key", "Player", "player", "Player_Clean", "name"],
    "team": ["Squad", "team", "club_tm"],
    "season": ["Season", "season"],
    "minutes": ["Min", "min", "Minutes"],
    "market_value": ["market_value_in_eur", "market_value", "Valor (€)", "Valor Mercado"],
    "npxg": ["NPxG_p90", "NPxG90", "NPxG/90"],
    "shots_on_target": ["SoT_p90", "SoT/90", "Tiros al arco/90"],
    "xa": ["xA_p90", "xA90", "xA/90"],
    "edad": ["edad", "Age", "Age_y", "Edad"],
    "goles": ["G_tot", "goals", "Gls", "Goles"],
    "asistencias": ["A_tot", "a", "Ast", "Asistencias"],
    "player_id": ["player_id", "id"],
}


def norm_text(value) -> str:
    value = unicodedata.normalize(
        "NFKD", str(value)
    ).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slug(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD", str(value)
    ).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    lookup = {norm_text(c): c for c in df.columns}
    for alias in aliases:
        key = norm_text(alias)
        if key in lookup:
            return lookup[key]
    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def euro(value) -> str:
    if pd.isna(value):
        return "Sin dato"

    value = float(value)

    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f} M"
    if value >= 1_000:
        return f"€{value / 1_000:.0f} k"

    return f"€{value:,.0f}"


def season_sort_key(value):
    text = str(value)
    m = re.search(r"(\d{2})-(\d{2})", text)

    if m:
        return int(m.group(1))

    return 999


# ============================================================
# FOTOS
# ============================================================

def local_photo(player: str) -> Optional[str]:
    if not FOTOS_DIR.exists():
        return None

    target = slug(player)

    candidates = list(FOTOS_DIR.glob("*"))

    for path in candidates:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        if slug(path.stem) == target:
            return str(path)

    return None


def transfermarkt_photo(player_id) -> Optional[str]:
    """
    Transfermarkt suele usar esta estructura para retratos.
    Se utiliza únicamente cuando el player_id existe.
    """
    if pd.isna(player_id):
        return None

    try:
        pid = int(float(player_id))
    except Exception:
        return None

    return f"https://img.a.transfermarkt.technology/portrait/big/{pid}-1686231459.jpg"


# ============================================================
# CARGA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"No se encontró el archivo principal:\n{DATA_PATH}"
        )
        st.stop()

    current = pd.read_csv(DATA_PATH, low_memory=False)

    if HIST_PATH.exists():
        history = pd.read_csv(HIST_PATH, low_memory=False)
    else:
        history = pd.DataFrame()

    return current, history


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    cols = {
        key: find_col(df, aliases)
        for key, aliases in ALIASES.items()
    }

    name_col = cols["name"]

    if name_col is None:
        st.error(
            "El CSV no contiene una columna reconocible para el nombre del jugador."
        )
        st.stop()

    df["Jugador"] = (
        df[name_col]
        .astype(str)
        .str.strip()
    )

    if cols["team"]:
        df["Equipo"] = (
            df[cols["team"]]
            .fillna("Sin equipo")
            .astype(str)
            .str.strip()
        )
    else:
        df["Equipo"] = "Sin equipo"

    for target in [
        "Minutos",
        "Valor (€)",
        "Edad",
        "Goles",
        "Asistencias",
    ]:
        source_key = {
            "Minutos": "minutes",
            "Valor (€)": "market_value",
            "Edad": "edad",
            "Goles": "goles",
            "Asistencias": "asistencias",
        }[target]

        source = cols[source_key]

        if source:
            df[target] = clean_numeric(df[source])
        else:
            df[target] = np.nan

    for target in ["npxg", "shots_on_target", "xa"]:
        source = cols[target]

        if source:
            df[target] = clean_numeric(df[source])
        else:
            df[target] = np.nan

    if cols["season"]:
        df["Temporada"] = df[cols["season"]].astype(str)
    else:
        df["Temporada"] = "Actual"

    if cols["player_id"]:
        df["player_id"] = df[cols["player_id"]]
    else:
        df["player_id"] = np.nan

    df["Perfil"] = "Centrodelantero"

    df = df[df["Jugador"].str.len() > 0].copy()

    return df.reset_index(drop=True)


# ============================================================
# CONSOLIDACIÓN
# ============================================================

def consolidate_current_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    La temporada 25-26 se utiliza para el ranking actual.
    Si no existe, se toma la temporada más reciente disponible.

    La consolidación evita mostrar varias filas del mismo jugador
    dentro del ranking actual.
    """
    seasons = df["Temporada"].dropna().astype(str).unique().tolist()

    current_season = "25-26" if "25-26" in seasons else None

    if current_season is None:
        if seasons:
            current_season = sorted(seasons, key=season_sort_key)[-1]
        else:
            current_season = None

    if current_season is not None:
        current = df[df["Temporada"] == current_season].copy()
    else:
        current = df.copy()

    # Si existen duplicados jugador-equipo, conservar el registro
    # con mayor cantidad de minutos.
    current = (
        current
        .sort_values("Minutos", ascending=False, na_position="last")
        .drop_duplicates(subset=["Jugador"], keep="first")
        .reset_index(drop=True)
    )

    current["Temporada ranking"] = current_season

    return current


# ============================================================
# AHP — wrapper to core.models
# ============================================================

def ahp_weights() -> np.ndarray:
    return _ahp_weights(
        np.array(SETTINGS["ahp"]["saaty_matrix"], dtype=float)
    )


# ============================================================
# IMPUTACIÓN — wrapper to core.models
# ============================================================

def impute_matrix(
    df: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    return _impute_matrix(df, columns)


# ============================================================
# TOPSIS — wrapper to core.models
# ============================================================

def topsis(
    X: np.ndarray,
    weights: np.ndarray,
    benefits: np.ndarray,
) -> np.ndarray:
    return _topsis(X, weights, benefits, normalize="vector")


# ============================================================
# MABAC — wrapper to core.models
# ============================================================

def mabac(
    X: np.ndarray,
    weights: np.ndarray,
    benefits: np.ndarray,
) -> np.ndarray:
    return _mabac(X, weights, benefits)


# ============================================================
# MEREC — wrapper to core.models
# ============================================================

def merec(
    X: np.ndarray,
    weights_unused=None,
    benefits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _merec(X, benefits=benefits)


# ============================================================
# RANKING
# ============================================================

CRITERIA_COLS = [
    "npxg",
    "shots_on_target",
    "xa",
    "Valor (€)",
]

CRITERIA_BENEFITS = np.array(
    [True, True, True, False],
    dtype=bool,
)

METHOD_KEYS = list(RANKING_METHODS)
METHOD_LABELS = {k: m.label for k, m in RANKING_METHODS.items()}


def rank_frame(
    df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    tw_alpha: float = 0.6,
    tw_beta: float = 0.4,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Ejecuta los 16 métodos de ranking sobre la misma matriz de decisión y
    con los mismos pesos (salvo MEREC, que calcula los suyos).
    Crea Puntaje_<MÉTODO> y Rank_<MÉTODO> (1 = mejor) para cada uno.
    """

    df = df.copy()

    X = impute_matrix(df, CRITERIA_COLS)
    benefits = CRITERIA_BENEFITS

    weights_used = (
        ahp_weights()
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    _, weights_merec = merec(X, benefits=benefits)

    # Desempate común: más NPxG y luego menor valor de mercado
    tie_npxg = -df["npxg"].fillna(-np.inf).to_numpy()
    tie_value = df["Valor (€)"].fillna(np.inf).to_numpy()

    for key, meth in RANKING_METHODS.items():

        if key == "THREE_WAY":
            tw = three_way(
                X,
                weights_used,
                benefits,
                alpha=tw_alpha,
                beta=tw_beta,
            )
            scores = tw.pc
            df["Decisión 3-Way"] = tw.decision
        else:
            scores = np.asarray(
                meth.run(X, weights_used, benefits),
                dtype=float,
            )

        df[f"Puntaje_{key}"] = scores

        primary = -scores if meth.higher_is_better else scores
        order = np.lexsort((tie_value, tie_npxg, primary))
        ranks = np.empty(len(df), dtype=int)
        ranks[order] = np.arange(1, len(df) + 1)
        df[f"Rank_{key}"] = ranks

    return df, weights_merec


def weight_sets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pesos de cada método de ponderación sobre los jugadores filtrados."""
    X = impute_matrix(df, CRITERIA_COLS)
    cfg = SETTINGS
    return {
        "AHP": ahp_weights(),
        "Entropía": entropy_weights(X),
        "MEREC": merec(X, benefits=CRITERIA_BENEFITS)[1],
        "SWARA": swara_weights(
            cfg["swara"]["order"],
            cfg["swara"]["s"],
        ),
        "ITARA": itara_weights(
            X,
            CRITERIA_BENEFITS,
            thresholds=float(cfg["itara"]["threshold"]),
        ),
        "BWM": bwm_weights(
            cfg["bwm"]["best_to_others"],
            cfg["bwm"]["others_to_worst"],
        ).weights,
        "ANP": anp_weights(
            np.array(cfg["anp"]["influence"], dtype=float)
        ).weights,
    }


# ============================================================
# MÉTRICAS EXTRA / EXTREMOS
# ============================================================

EXTRA_ALIASES = {
    "Regates": [
        "Succ",
        "Take-Ons",
        "Take Ons",
        "Dribbles",
        "dribbles",
    ],
    "Regates exitosos": [
        "Succ%",
        "Take-On%",
        "Take On%",
    ],
    "Progresiones": [
        "PrgC",
        "Progressive Carries",
        "Carries",
    ],
    "Pases progresivos": [
        "PrgP",
        "Progressive Passes",
    ],
    "Acciones de tiro": [
        "Att",
        "Att Pen",
        "Attacking Actions",
    ],
}


def add_extra_metrics(
    current: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:

    current = current.copy()

    for target, aliases in EXTRA_ALIASES.items():

        source = find_col(raw, aliases)

        if source is None:
            continue

        mapping = raw[
            [
                find_col(raw, ALIASES["name"]),
                source,
            ]
        ].copy()

        mapping.columns = [
            "Jugador",
            target,
        ]

        mapping[target] = clean_numeric(
            mapping[target]
        )

        mapping = (
            mapping
            .dropna(subset=["Jugador"])
            .drop_duplicates(
                "Jugador",
                keep="first",
            )
        )

        current = current.merge(
            mapping,
            on="Jugador",
            how="left",
        )

    return current


# ============================================================
# EQUIPOS
# ============================================================

TEAM_LOGOS = {
    "arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "manchester city": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "manchester united": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "manchester utd": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "aston villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_logo.svg",
    "newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "west ham": "https://upload.wikimedia.org/wikipedia/en/e/e0/West_Ham_United_FC.svg",
    "real madrid": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg",
    "barcelona": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg",
    "atletico madrid": "https://upload.wikimedia.org/wikipedia/en/f/f4/Atletico_Madrid_2017_logo.svg",
    "bayern": "https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg",
    "borussia dortmund": "https://upload.wikimedia.org/wikipedia/commons/6/67/Borussia_Dortmund_logo.svg",
    "leverkusen": "https://upload.wikimedia.org/wikipedia/en/5/59/Bayer_04_Leverkusen_logo.svg",
    "leipzig": "https://upload.wikimedia.org/wikipedia/en/0/04/RB_Leipzig_2014_logo.svg",
    "paris sg": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg",
    "monaco": "https://upload.wikimedia.org/wikipedia/en/b/ba/AS_Monaco_FC.svg",
    "juventus": "https://upload.wikimedia.org/wikipedia/commons/b/bc/Juventus_FC_2017_icon_%28black%29.svg",
    "inter": "https://upload.wikimedia.org/wikipedia/commons/0/05/FC_Internazionale_Milano_2021.svg",
    "milan": "https://upload.wikimedia.org/wikipedia/commons/d/d0/Logo_of_AC_Milan.svg",
    "napoli": "https://upload.wikimedia.org/wikipedia/commons/2/28/SSC_Napoli_Logo_2023.svg",
    "roma": "https://upload.wikimedia.org/wikipedia/en/f/f7/AS_Roma_logo_%282017%29.svg",
    "lazio": "https://upload.wikimedia.org/wikipedia/en/e/e4/SS_Lazio.svg",
}


def get_team_logo(team: str) -> Optional[str]:
    normalized = norm_text(team)

    for key, url in TEAM_LOGOS.items():
        if norm_text(key) in normalized:
            return url

    return None


# ============================================================
# HISTORIAL
# ============================================================

def player_history(
    history: pd.DataFrame,
    player: str,
) -> pd.DataFrame:

    if history.empty:
        return pd.DataFrame()

    name_col = find_col(
        history,
        ALIASES["name"],
    )

    if name_col is None:
        return pd.DataFrame()

    h = history.copy()

    h["_name_norm"] = (
        h[name_col]
        .astype(str)
        .map(norm_text)
    )

    target = norm_text(player)

    h = h[
        h["_name_norm"] == target
    ].copy()

    if h.empty:
        return h

    season_col = find_col(
        h,
        ALIASES["season"],
    )

    if season_col:
        h["_season_sort"] = h[season_col].map(
            season_sort_key
        )
        h = h.sort_values("_season_sort")

    return h


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

raw_df, raw_hist = load_data()

df_all = prepare_data(raw_df)

current = consolidate_current_players(
    df_all
)

current = add_extra_metrics(
    current,
    raw_df,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Filtros de scouting")

    st.info(
        "Ranking actual basado en la temporada 25/26 "
        "cuando está disponible."
    )

    min_minutes = st.slider(
        "Minutos mínimos",
        min_value=0,
        max_value=4500,
        value=900,
        step=100,
    )

    max_value_data = current[
        "Valor (€)"
    ].dropna()

    max_value = (
        int(max_value_data.max())
        if not max_value_data.empty
        else 200_000_000
    )

    budget = st.slider(
        "Presupuesto máximo",
        min_value=0,
        max_value=max_value,
        value=max_value,
        step=max(1_000_000, max_value // 100),
        format="€%d",
    )

    method = st.selectbox(
        "Modelo de ranking",
        METHOD_KEYS,
        format_func=lambda k: (
            METHOD_LABELS[k]
            + ("" if RANKING_METHODS[k].higher_is_better else " (menor = mejor)")
        ),
    )

    weight_method = st.selectbox(
        "Método de pesos",
        WEIGHT_METHODS,
        index=WEIGHT_METHODS.index("Manual (sliders)"),
        help=(
            "Manual: sliders que arrancan en AHP. "
            "AHP / SWARA / BWM / ANP: pesos subjetivos definidos en "
            "config/settings.yaml. Entropía / MEREC / ITARA: pesos "
            "objetivos calculados con los jugadores filtrados."
        ),
    )

    top_n = st.slider(
        "Jugadores en ranking",
        5,
        min(30, max(5, len(current))),
        min(15, max(5, len(current))),
    )

    gems_min_minutes = st.slider(
        "Mínimo de minutos (gemas)",
        min_value=0,
        max_value=3000,
        value=900,
        step=50,
    )

    with st.expander("Parámetros Three-Way"):
        tw_alpha = st.slider(
            "α (aceptar si PC ≥ α)",
            0.05, 1.0,
            float(SETTINGS["three_way"]["alpha"]),
            0.05,
        )
        tw_beta = st.slider(
            "β (rechazar si PC ≤ β)",
            0.0, 0.95,
            float(SETTINGS["three_way"]["beta"]),
            0.05,
        )
        if tw_beta >= tw_alpha:
            st.warning("β debe ser menor que α; se ajusta β = α − 0.05.")
            tw_beta = max(0.0, tw_alpha - 0.05)

    with st.expander("Métodos disponibles"):
        st.markdown(
            "\n\n".join(
                f"**{m.label}**"
                + ("" if m.higher_is_better else " · _menor = mejor_")
                + f": {m.description}"
                for m in RANKING_METHODS.values()
            )
            + "\n\n---\n\n"
            "**Pesos:** AHP, Entropía, MEREC (ramo, 1ª prueba) · "
            "SWARA, ITARA, BWM, ANP (2ª prueba)."
        )

    default_weights = ahp_weights()
    default_percents = [
        int(round(w * 100)) for w in default_weights
    ]

    slider_percents = default_percents

    if weight_method == "Manual (sliders)":

        # Pesos por criterio: sliders 0-100 (arrancan en los pesos AHP)
        reset_weights = st.button("Restablecer a AHP")

        if reset_weights:
            for key, pct in zip(AHP_SLIDER_KEYS, default_percents):
                st.session_state[key] = pct

        slider_percents = []
        for key, criterion, pct in zip(
            AHP_SLIDER_KEYS,
            AHP_CRITERIA,
            default_percents,
        ):
            value = st.slider(
                f"{criterion}",
                min_value=0,
                max_value=100,
                value=st.session_state.get(key, pct),
                key=key,
            )
            slider_percents.append(value)

    total = sum(slider_percents)

    if total == 0:
        manual_weights = default_weights
        st.warning(
            "Todos los pesos están en 0: se usan los pesos AHP "
            "por defecto."
        )
    else:
        manual_weights = (
            np.array(slider_percents, dtype=float) / total
        )


# ============================================================
# FILTRADO
# ============================================================

filtered = current.copy()

filtered = filtered[
    filtered["Minutos"].fillna(0)
    >= min_minutes
]

filtered = filtered[
    filtered["Valor (€)"].isna()
    | (
        filtered["Valor (€)"]
        <= budget
    )
]

if filtered.empty:
    st.warning(
        "No quedan jugadores con los filtros seleccionados."
    )
    st.stop()


# ============================================================
# RANKING RECALCULADO
# ============================================================

all_weight_sets = weight_sets(filtered)

effective_weights = (
    manual_weights
    if weight_method == "Manual (sliders)"
    else all_weight_sets[weight_method]
)

with st.sidebar:
    st.markdown(
        f"""
<div class="sidebar-box">
<h4>Pesos efectivos</h4>
<p class="sidebar-meta">{weight_method}</p>
""" +
"".join(
    f'<div class="weight-row"><span>{c} <small>{"maximizar" if b else "minimizar"}</small></span>'
    f'<span class="measured">{w:.1%}</span></div>\n'
    for c, w, b in zip(
        AHP_CRITERIA,
        effective_weights,
        AHP_BENEFITS,
    )
)
        + """</div>
""",
        unsafe_allow_html=True,
    )

ranked, merec_weights = rank_frame(
    filtered,
    weights=effective_weights,
    tw_alpha=tw_alpha,
    tw_beta=tw_beta,
)
method_meta = RANKING_METHODS[method]
method_label = METHOD_LABELS[method]

rank_col = f"Rank_{method}"
score_col = f"Puntaje_{method}"

ranked = ranked.sort_values(
    rank_col
).reset_index(drop=True)

# Índice 0–1 (1 = mejor) para colorear igual en métodos "mayor = mejor"
# y "menor = mejor" (VIKOR, MAIRCA).
ranked["Índice relativo"] = 1 - (
    (ranked[rank_col] - 1) / max(len(ranked) - 1, 1)
)


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    """
<div class="hero">
<h1 class="hero-title">Scouting Lab</h1>
<p class="hero-copy">
Modelo multicriterio para identificar delanteros centro con
alta producción ofensiva, coste razonable y potencial de
mercado en las principales ligas europeas. Data-driven
recruitment, al estilo Moneyball.
</p>
</div>
""",
    unsafe_allow_html=True,
)

selected_name = st.selectbox(
    "Busca y selecciona un delantero",
    ranked["Jugador"].tolist(),
)

profile = ranked[
    ranked["Jugador"] == selected_name
].iloc[0]

team_logo = get_team_logo(
    profile["Equipo"]
)

if team_logo:
    st.markdown(
        f"""
<img src="{team_logo}"
style="
position:fixed;
top:50%;
left:55%;
transform:translate(-50%,-50%);
width:43vw;
max-width:550px;
opacity:.045;
z-index:0;
pointer-events:none;
">
""",
        unsafe_allow_html=True,
    )




# ============================================================
# KPIs
# ============================================================

st.markdown(
    '<h2 class="section-head">Resumen del mercado</h2>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Delanteros analizados",
    f"{len(ranked):,}",
)

k2.metric(
    "Equipos",
    f"{ranked['Equipo'].nunique():,}",
)

k3.metric(
    "Valor mediano",
    euro(ranked["Valor (€)"].median()),
)

k4.metric(
    "Edad media",
    (
        f"{ranked['Edad'].mean():.1f} años"
        if ranked["Edad"].notna().any()
        else "N/A"
    ),
)

k5.metric(
    "Modelo activo",
    method_label,
    help=f"Pesos: {weight_method}",
)


# ============================================================
# RANKING + MONEYBALL
# ============================================================

left, right = st.columns([1.25, 1])

with left:

    st.markdown(
        '<h2 class="section-head">Ranking del mercado</h2>',
        unsafe_allow_html=True,
    )

    top = ranked.head(top_n).copy()

    fig_rank = px.bar(
        top.sort_values(
            rank_col,
            ascending=False,
        ),
        x=score_col,
        y="Jugador",
        orientation="h",
        color="Índice relativo",
        hover_data={
            "Equipo": True,
            rank_col: True,
            score_col: ":.4f",
            "Valor (€)": True,
            "npxg": ":.2f",
            "Índice relativo": False,
        },
        color_continuous_scale=SCALE_COOL,
        template=CHART_TEMPLATE,
    )

    if not method_meta.higher_is_better:
        st.caption(
            f"En {method_label} un score MENOR es mejor: "
            "las barras más cortas son los mejores jugadores."
        )

    fig_rank.update_layout(
        height=500,
        margin=dict(
            l=10,
            r=15,
            t=20,
            b=15,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        fig_rank,
        use_container_width=True,
    )


with right:

    st.markdown(
        '<h2 class="section-head">Moneyball: rendimiento vs coste</h2>',
        unsafe_allow_html=True,
    )

    scatter = ranked.copy()

    scatter["Valor visual (€M)"] = (
        scatter["Valor (€)"] / 1_000_000
    )

    scatter["Tamaño"] = (
        scatter["Minutos"]
        .fillna(0)
        .clip(lower=1)
    )

    fig_scatter = px.scatter(
        scatter,
        x="Valor visual (€M)",
        y="npxg",
        size="Tamaño",
        color="Índice relativo",
        hover_name="Jugador",
        hover_data={
            "Equipo": True,
            "Valor visual (€M)": ":.2f",
            "npxg": ":.2f",
            "shots_on_target": ":.2f",
            "xa": ":.2f",
            score_col: ":.4f",
            "Índice relativo": False,
        },
        color_continuous_scale=SCALE_COOL,
        template=CHART_TEMPLATE,
    )

    fig_scatter.update_layout(
        height=500,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=15,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )


# ============================================================
# GEMAS OCULTAS
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Gemas ocultas: bajo coste, alta producción</h2>',
    unsafe_allow_html=True,
)

_gems_cols = ["Valor (€)", "npxg", "Minutos"]

_gems_valid = ranked[
    ranked["Valor (€)"].notna()
    & ranked["npxg"].notna()
    & ranked["Minutos"].notna()
].copy()

_gems_lowmin = _gems_valid[
    _gems_valid["Minutos"] < gems_min_minutes
]

gems = _gems_valid[
    _gems_valid["Minutos"] >= gems_min_minutes
].copy()

_gems_missing = len(ranked) - len(_gems_valid)
_excluded_lowmin = len(_gems_lowmin)

if len(ranked) - len(gems) > 0:
    st.caption(
        f"Quedaron fuera {len(ranked) - len(gems)} jugadores: "
        f"{_excluded_lowmin} con menos de {gems_min_minutes}' "
        f"y {_gems_missing} sin datos completos."
    )

if gems.empty:
    st.warning(
        "No quedan jugadores con los filtros seleccionados "
        "para el análisis de gemas ocultas."
    )
else:
    gems["Valor (€M)"] = gems["Valor (€)"] / 1_000_000

    med_cost = gems["Valor (€M)"].median()
    med_npxg = gems["npxg"].median()

    gems["Es gema"] = (
        (gems["Valor (€M)"] < med_cost)
        & (gems["npxg"] > med_npxg)
    )

    n_gems = int(gems["Es gema"].sum())

    color_map = {
        True: PALETTE["accent"],
        False: PALETTE["rule_2"],
    }

    fig_gems = px.scatter(
        gems,
        x="Valor (€M)",
        y="npxg",
        size="Minutos",
        color="Es gema",
        color_discrete_map=color_map,
        hover_name="Jugador",
        hover_data={
            "Equipo": True,
            "Valor (€M)": ":.1f",
            "npxg": ":.2f",
            "Minutos": ":.0f",
            "Es gema": False,
        },
        log_x=True,
        template=CHART_TEMPLATE,
    )

    fig_gems.add_vline(
        x=med_cost,
        line_dash="dot",
        line_color=PALETTE["muted"],
        annotation_text=f"Mediana coste €{med_cost:.0f}M",
        annotation_position="top left",
    )
    fig_gems.add_hline(
        y=med_npxg,
        line_dash="dot",
        line_color=PALETTE["muted"],
        annotation_text=f"Mediana NPxG {med_npxg:.2f}",
        annotation_position="bottom right",
    )

    fig_gems.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color=PALETTE["paper"]),
        ),
        selector=dict(mode="markers"),
    )

    top_gems = gems[gems["Es gema"]].nlargest(
        10, "npxg"
    )
    for _, row in top_gems.iterrows():
        fig_gems.add_annotation(
            x=row["Valor (€M)"],
            y=row["npxg"],
            text=row["Jugador"],
            showarrow=True,
            arrowhead=2,
            arrowcolor=PALETTE["accent"],
            arrowsize=0.7,
            arrowwidth=1,
            font=dict(size=11, color=PALETTE["accent"]),
            bgcolor=PALETTE["paper_3"],
            bordercolor=PALETTE["rule_2"],
            borderwidth=1,
            borderpad=2,
        )

    fig_gems.update_layout(
        height=560,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=15,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Valor de mercado (€M) · escala log",
            gridcolor=PALETTE["rule"],
            # Rango fijo: el autorange de ejes log con shapes (add_vline)
            # dispara un bug de plotly.js que estira el eje hasta 10^12.
            # Con type="log", range va en EXPONENTES (log10 del valor).
            range=[
                float(np.log10(gems["Valor (€M)"].min() * 0.8)),
                float(np.log10(gems["Valor (€M)"].max() * 1.2)),
            ],
        ),
        yaxis=dict(
            title="NPxG/90",
            gridcolor=PALETTE["rule"],
        ),
        legend=dict(
            title_text="",
            itemsizing="constant",
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig_gems,
        use_container_width=True,
    )

    st.caption(
        f"{n_gems} gemas ocultas de {len(gems)} jugadores "
        f"({n_gems / len(gems):.0%}): coste por debajo de la mediana "
        f"(€{med_cost:.0f}M) y NPxG/90 por encima ({med_npxg:.2f})."
    )


# ============================================================
# COMPARACIÓN DE MÉTODOS MCDM
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Comparación de métodos MCDM</h2>',
    unsafe_allow_html=True,
)

st.caption(
    f"Los {len(METHOD_KEYS)} métodos de ranking usan la misma matriz "
    f"(NPxG/90, tiros al arco/90, xA/90, valor de mercado) y los pesos "
    f"**{weight_method}**, salvo MEREC, que calcula sus propios pesos."
)

selected_methods = st.multiselect(
    "Métodos a comparar",
    METHOD_KEYS,
    default=METHOD_KEYS,
    format_func=lambda k: METHOD_LABELS[k],
)

if len(selected_methods) < 2:
    st.info("Selecciona al menos dos métodos para compararlos.")
else:
    rank_cols = [f"Rank_{k}" for k in selected_methods]

    consensus = ranked[["Jugador", "Equipo", "Valor (€)"] + rank_cols].copy()
    consensus["Ranking promedio"] = consensus[rank_cols].mean(axis=1)
    consensus["Mejor posición"] = consensus[rank_cols].min(axis=1)
    consensus["Peor posición"] = consensus[rank_cols].max(axis=1)
    consensus["Veces en Top 10"] = (consensus[rank_cols] <= 10).sum(axis=1)
    consensus = consensus.sort_values(
        ["Ranking promedio", "Veces en Top 10"],
        ascending=[True, False],
    )
    top_consensus = consensus.head(top_n)

    tab_cons, tab_heat, tab_corr, tab_3w = st.tabs(
        [
            "Consenso",
            "Posición por método",
            "Concordancia entre métodos",
            "Decisión Three-Way",
        ]
    )

    with tab_cons:

        fig_consensus = go.Figure()

        plot_cons = top_consensus.iloc[::-1]

        fig_consensus.add_trace(
            go.Bar(
                x=plot_cons["Peor posición"] - plot_cons["Mejor posición"],
                base=plot_cons["Mejor posición"],
                y=plot_cons["Jugador"],
                orientation="h",
                marker_color=PALETTE["cool_soft"],
                name="Rango (mejor–peor)",
                hovertemplate="Mejor #%{base}<extra></extra>",
            )
        )

        fig_consensus.add_trace(
            go.Scatter(
                x=plot_cons["Ranking promedio"],
                y=plot_cons["Jugador"],
                mode="markers",
                marker=dict(size=12, color=PALETTE["cool"]),
                name="Ranking promedio",
                customdata=plot_cons[["Veces en Top 10"]],
                hovertemplate=(
                    "Promedio %{x:.1f}<br>"
                    "Top 10 en %{customdata[0]} métodos<extra></extra>"
                ),
            )
        )

        fig_consensus.update_layout(
            template=CHART_TEMPLATE,
            height=max(400, top_n * 27),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Posición en el ranking (1 = mejor)",
            legend=dict(orientation="h", y=1.06),
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig_consensus, use_container_width=True)

        st.caption(
            "Punto = ranking promedio entre los métodos elegidos; barra = "
            "rango entre su mejor y su peor posición. Barras cortas indican "
            "un jugador robusto: todos los métodos coinciden."
        )

    with tab_heat:

        heat = top_consensus.set_index("Jugador")[rank_cols]
        heat.columns = [METHOD_LABELS[k] for k in selected_methods]

        fig_heat = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=SCALE_COOL_REV,
            zmin=1,
            zmax=max(30, top_n * 2),
            template=CHART_TEMPLATE,
        )

        fig_heat.update_layout(
            height=max(400, top_n * 28),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar_title="Rank",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig_heat.update_xaxes(side="top")

        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            "Posición de los jugadores del consenso en cada método "
            "(verde = arriba en el ranking)."
        )

    with tab_corr:

        corr = ranked[rank_cols].corr(method="spearman")
        corr.index = corr.columns = [
            METHOD_LABELS[k] for k in selected_methods
        ]

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale=[PALETTE["bad"], PALETTE["paper_2"], PALETTE["cool"]],
            template=CHART_TEMPLATE,
        )

        fig_corr.update_layout(
            height=620,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar_title="ρ",
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()

        if not off_diag.empty:
            hi_pair = off_diag.idxmax()
            lo_pair = off_diag.idxmin()
            mean_rho = corr.where(
                ~np.eye(len(corr), dtype=bool)
            ).stack().mean()
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("ρ de Spearman promedio", f"{mean_rho:.2f}")
            cc2.metric(
                "Más parecidos",
                f"{hi_pair[0]} · {hi_pair[1]}",
                f"ρ = {off_diag.max():.2f}",
                delta_color="off",
            )
            cc3.metric(
                "Más distintos",
                f"{lo_pair[0]} · {lo_pair[1]}",
                f"ρ = {off_diag.min():.2f}",
                delta_color="off",
            )

        st.caption(
            "Correlación de Spearman entre los rankings completos: 1 = "
            "mismo orden, 0 = sin relación. Métodos que penalizan fuerte "
            "el costo (COPRAS) o con pesos propios (MEREC) suelen separarse "
            "del resto."
        )

    with tab_3w:

        tw_counts = ranked["Decisión 3-Way"].value_counts()

        t1, t2, t3 = st.columns(3)
        t1.metric("Aceptar", int(tw_counts.get("Aceptar", 0)))
        t2.metric("Frontera", int(tw_counts.get("Frontera", 0)))
        t3.metric("Rechazar", int(tw_counts.get("Rechazar", 0)))

        fig_tw = px.histogram(
            ranked,
            x="Puntaje_THREE_WAY",
            color="Decisión 3-Way",
            nbins=40,
            color_discrete_map={
                "Aceptar": PALETTE["ok"],
                "Frontera": PALETTE["accent"],
                "Rechazar": PALETTE["bad"],
            },
            category_orders={
                "Decisión 3-Way": ["Aceptar", "Frontera", "Rechazar"]
            },
            template=CHART_TEMPLATE,
        )
        fig_tw.add_vline(x=tw_alpha, line_dash="dot", line_color=PALETTE["ok"],
                         annotation_text=f"α = {tw_alpha:.2f}")
        fig_tw.add_vline(x=tw_beta, line_dash="dot", line_color=PALETTE["bad"],
                         annotation_text=f"β = {tw_beta:.2f}",
                         annotation_position="top left")
        fig_tw.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Probabilidad compuesta PC = Σ w·N",
            yaxis_title="Jugadores",
            legend_title_text="",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_tw, use_container_width=True)

        accepted = ranked[ranked["Decisión 3-Way"] == "Aceptar"].sort_values(
            "Puntaje_THREE_WAY", ascending=False
        )
        if not accepted.empty:
            st.markdown("**Shortlist aceptada**")
            st.dataframe(
                accepted[["Jugador", "Equipo", "Valor (€)", "npxg",
                          "Puntaje_THREE_WAY"]]
                .rename(columns={"npxg": "NPxG/90",
                                 "Puntaje_THREE_WAY": "PC"})
                .head(25),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# PESOS
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Comparación de métodos de pesos</h2>',
    unsafe_allow_html=True,
)

weights_long = pd.DataFrame(
    [
        {"Método": name, "Criterio": crit, "Peso": float(w)}
        for name, ws in (
            {"Manual (sliders)": manual_weights} | all_weight_sets
        ).items()
        for crit, w in zip(AHP_CRITERIA, ws)
    ]
)

fig_weights = px.bar(
    weights_long,
    x="Método",
    y="Peso",
    color="Criterio",
    text=weights_long["Peso"].map(lambda x: f"{x:.0%}"),
    color_discrete_sequence=[PALETTE["cool"], PALETTE["accent"], PALETTE["ink_2"], PALETTE["muted"]],
    template=CHART_TEMPLATE,
)

fig_weights.update_layout(
    barmode="stack",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis_tickformat=".0%",
    legend_title_text="",
    legend=dict(orientation="h", y=1.08),
    margin=dict(l=10, r=10, t=40, b=10),
)

fig_weights.update_traces(textposition="inside")

st.plotly_chart(fig_weights, use_container_width=True)

ahp_cons = ahp_consistency(np.array(SETTINGS["ahp"]["saaty_matrix"], dtype=float))
bwm_res = bwm_weights(
    SETTINGS["bwm"]["best_to_others"],
    SETTINGS["bwm"]["others_to_worst"],
)
anp_res = anp_weights(np.array(SETTINGS["anp"]["influence"], dtype=float))

w1, w2, w3, w4 = st.columns(4)
w1.metric("AHP · CR", f"{ahp_cons['CR']:.3f}",
          "consistente" if ahp_cons["CR"] < 0.1 else "revisar",
          delta_color="off")
w2.metric("BWM · CR", f"{bwm_res.consistency_ratio:.3f}",
          f"ξ = {bwm_res.xi:.2f}", delta_color="off")
w3.metric("ANP · supermatriz límite",
          f"{anp_res.iterations} iteraciones", delta_color="off")
w4.metric("Pesos en uso", weight_method)

st.caption(
    "Subjetivos (juicio experto, editables en config/settings.yaml): AHP, "
    "SWARA, BWM, ANP. Objetivos (salen de los datos filtrados): Entropía, "
    "MEREC, ITARA. Cambia el método en la barra lateral para ver cómo se "
    "mueve el ranking."
)


# ============================================================
# PLAYER INTELLIGENCE
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Player Intelligence</h2>',
    unsafe_allow_html=True,
)

history = player_history(
    raw_hist,
    selected_name,
)


p1, p2, p3 = st.columns(
    [1, 1.45, 1.1]
)


with p1:

    photo = local_photo(
        profile["Jugador"]
    )

    if photo:
        st.image(
            photo,
            use_container_width=True,
        )
    else:
        remote_photo = transfermarkt_photo(
            profile.get("player_id")
        )

        if remote_photo:
            st.image(
                remote_photo,
                use_container_width=True,
            )
        else:
            st.markdown(
                """
<div class="panel photo-empty">
Sin foto disponible
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
<div class="player-card">
<h2>
{profile["Jugador"].title()}
</h2>

<span class="badge">
{profile["Equipo"]}
</span>
</div>
""",
        unsafe_allow_html=True,
    )


with p2:

    st.markdown(
        "#### Estadísticas de temporada"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Goles",
        (
            int(profile["Goles"])
            if pd.notna(profile["Goles"])
            else "-"
        ),
    )

    a2.metric(
        "Asistencias",
        (
            int(profile["Asistencias"])
            if pd.notna(profile["Asistencias"])
            else "-"
        ),
    )

    a3.metric(
        "Minutos",
        (
            f"{int(profile['Minutos'])}'"
            if pd.notna(profile["Minutos"])
            else "-"
        ),
    )

    st.markdown(
        "#### Métricas predictivas /90"
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "NPxG",
        (
            f"{profile['npxg']:.2f}"
            if pd.notna(profile["npxg"])
            else "-"
        ),
    )

    b2.metric(
        "Tiros al arco",
        (
            f"{profile['shots_on_target']:.2f}"
            if pd.notna(
                profile["shots_on_target"]
            )
            else "-"
        ),
    )

    b3.metric(
        "xA",
        (
            f"{profile['xa']:.2f}"
            if pd.notna(profile["xa"])
            else "-"
        ),
    )


with p3:

    st.markdown(
        f"#### Algoritmo {method_label}"
    )

    st.metric(
        "Ranking",
        f"#{int(profile[rank_col])} / {len(ranked)}",
    )

    st.metric(
        "Score",
        f"{float(profile[score_col]):.4f}",
    )

    st.metric(
        "Valor de mercado",
        euro(profile["Valor (€)"]),
    )

    if pd.notna(profile["Edad"]):
        st.metric(
            "Edad",
            f"{int(profile['Edad'])} años",
        )

    st.markdown(
        f"**Decisión Three-Way:** {profile['Decisión 3-Way']}"
    )


# Posición del jugador en todos los métodos
player_ranks = pd.DataFrame(
    {
        "Método": [METHOD_LABELS[k] for k in METHOD_KEYS],
        "Posición": [int(profile[f"Rank_{k}"]) for k in METHOD_KEYS],
    }
).sort_values("Posición")

fig_player_ranks = px.bar(
    player_ranks,
    x="Método",
    y="Posición",
    text="Posición",
    color="Posición",
    color_continuous_scale=SCALE_COOL_REV,
    template=CHART_TEMPLATE,
)
fig_player_ranks.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    coloraxis_showscale=False,
    yaxis=dict(autorange="reversed", title="Posición (1 = mejor)"),
    xaxis_title="",
    margin=dict(l=10, r=10, t=30, b=10),
    title=dict(
        text=f"Posición de {profile['Jugador'].title()} en los "
        f"{len(METHOD_KEYS)} métodos (de {len(ranked)})",
        font=dict(size=14),
    ),
)
st.plotly_chart(fig_player_ranks, use_container_width=True)


# ============================================================
# EXTREMOS / MÉTRICAS ADICIONALES
# ============================================================

extra_available = [
    col
    for col in [
        "Regates",
        "Regates exitosos",
        "Progresiones",
        "Pases progresivos",
        "Acciones de tiro",
    ]
    if col in profile.index
    and pd.notna(profile[col])
]

if extra_available:

    st.markdown("---")

    st.markdown(
        '<h2 class="section-head">Perfil de juego y acciones ofensivas</h2>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(
        min(5, len(extra_available))
    )

    for container, metric in zip(
        metric_cols,
        extra_available,
    ):
        value = profile[metric]

        if "%" in metric:
            display_value = f"{float(value):.1f}%"
        else:
            display_value = f"{float(value):.2f}"

        container.metric(
            metric,
            display_value,
        )

else:

    st.caption(
        "Las métricas de regates/progresiones no están disponibles "
        "en el CSV actual; no se inventan valores."
    )


# ============================================================
# HISTORIAL 5-6 TEMPORADAS
# ============================================================

if not history.empty:

    st.markdown("---")

    st.markdown(
        '<h2 class="section-head">Evolución histórica</h2>',
        unsafe_allow_html=True,
    )

    season_col = find_col(
        history,
        ALIASES["season"],
    )

    goals_col = find_col(
        history,
        [
            "G_tot",
            "goals",
            "Gls",
        ],
    )

    npxg_col = find_col(
        history,
        [
            "NPxG90",
            "NPxG_p90",
        ],
    )

    if season_col and (
        goals_col or npxg_col
    ):

        hist_plot = history.copy()

        hist_plot["_sort"] = (
            hist_plot[season_col]
            .map(season_sort_key)
        )

        hist_plot = hist_plot.sort_values(
            "_sort"
        ).tail(6)

        fig_hist = go.Figure()

        if goals_col:
            fig_hist.add_trace(
                go.Bar(
                    x=hist_plot[season_col],
                    y=clean_numeric(
                        hist_plot[goals_col]
                    ),
                    name="Goles",
                )
            )

        if npxg_col:
            fig_hist.add_trace(
                go.Scatter(
                    x=hist_plot[season_col],
                    y=clean_numeric(
                        hist_plot[npxg_col]
                    ),
                    name="NPxG/90",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(width=3),
                )
            )

        fig_hist.update_layout(
            template=CHART_TEMPLATE,
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
            yaxis=dict(
                title="Goles",
            ),
            yaxis2=dict(
                title="NPxG/90",
                overlaying="y",
                side="right",
            ),
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )


# ============================================================
# COMPARACIÓN DEL JUGADOR CON EL MERCADO
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Radar de rendimiento</h2>',
    unsafe_allow_html=True,
)

radar_metrics = [
    "npxg",
    "shots_on_target",
    "xa",
]

labels = [
    "NPxG/90",
    "Tiros al arco/90",
    "xA/90",
]

market_values = []

for col in radar_metrics:

    series = ranked[col].astype(float)

    lo = series.min()
    hi = series.max()

    value = float(profile[col])

    if pd.isna(value):
        normalized = 0
    elif np.isclose(lo, hi):
        normalized = 50
    else:
        normalized = (
            (value - lo)
            / (hi - lo)
            * 100
        )

    market_values.append(
        normalized
    )

fig_radar = go.Figure()

fig_radar.add_trace(
    go.Scatterpolar(
        r=market_values + [market_values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=profile["Jugador"],
    )
)

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
        )
    ),
    template=CHART_TEMPLATE,
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(
    fig_radar,
    use_container_width=True,
)


# ============================================================
# DATA EXPLORER
# ============================================================

st.markdown("---")

st.markdown(
    '<h2 class="section-head">Data Explorer: ranking completo</h2>',
    unsafe_allow_html=True,
)

show_cols = [
    rank_col,
    "Jugador",
    "Edad",
    "Equipo",
    "Minutos",
    "Goles",
    "Valor (€)",
    "npxg",
    "shots_on_target",
    "xa",
    "Decisión 3-Way",
] + [f"Rank_{k}" for k in METHOD_KEYS if f"Rank_{k}" != rank_col] + [
    f"Puntaje_{k}" for k in METHOD_KEYS
]

table = ranked[
    [
        c
        for c in show_cols
        if c in ranked.columns
    ]
].rename(
    columns={
        "Valor (€)": "Valor Mercado",
        "npxg": "NPxG/90",
        "shots_on_target": "Tiros al arco/90",
        "xa": "xA/90",
    }
    | {f"Rank_{k}": f"Rank {METHOD_LABELS[k]}" for k in METHOD_KEYS}
    | {f"Puntaje_{k}": f"Score {METHOD_LABELS[k]}" for k in METHOD_KEYS}
)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DESCARGA
# ============================================================

csv_download = ranked.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Descargar ranking actual",
    data=csv_download,
    file_name="ranking_scouting_actual.csv",
    mime="text/csv",
)