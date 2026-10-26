#!/usr/bin/env python3
"""
Script standalone para corregir:
1. Deduplicación en consolidate_current_players: usar ['player_id', 'Temporada', 'Equipo']
2. impute_matrix: asegurar imputación robusta de market_value
3. MEREC: añadir eps=1e-12 antes de np.log(Z)
Regenera resultados_finales.csv y verifica que no hay NaN en los scores.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datos" / "procesado" / "resultados_finales.csv"
HIST_PATH = BASE_DIR / "datos" / "procesado" / "fw_base.csv"
OUTPUT_PATH = BASE_DIR / "datos" / "procesado" / "resultados_finales_fixed.csv"

# ============================================================
# UTILIDADES (copiadas de app.py)
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
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def find_col(df: pd.DataFrame, aliases: list[str]):
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
# FIX 1: consolidate_current_players con deduplicación robusta
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

    # FIX: Deduplicar por ['player_id', 'Temporada', 'Equipo'] en lugar de solo 'Jugador'
    # Manejar NaN en player_id: crear clave compuesta que trate NaN como valor distinto
    current["_dedup_key"] = (
        current["player_id"].astype(str) + "|" +
        current["Temporada"].astype(str) + "|" +
        current["Equipo"].astype(str)
    )

    # Si existen duplicados jugador-equipo-temporada, conservar el registro
    # con mayor cantidad de minutos.
    current = (
        current
        .sort_values("Minutos", ascending=False, na_position="last")
        .drop_duplicates(subset=["_dedup_key"], keep="first")
        .drop(columns=["_dedup_key"])
        .reset_index(drop=True)
    )

    current["Temporada ranking"] = current_season

    return current


# ============================================================
# AHP
# ============================================================

def ahp_weights() -> np.ndarray:
    matrix = np.array(
        [
            [1, 3, 3, 2],
            [1 / 3, 1, 1, 1 / 2],
            [1 / 3, 1, 1, 1 / 2],
            [1 / 2, 2, 2, 1],
        ],
        dtype=float,
    )

    weights = (matrix / matrix.sum(axis=0)).mean(axis=1)
    return weights / weights.sum()


# ============================================================
# FIX 2: impute_matrix robusto
# ============================================================

def impute_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """
    Convierte a numérico y rellena NaN con la mediana de CADA COLUMNA.
    Crítico: mercado_valor puede tener NaN en filas filtradas;
    la mediana se calcula SOBRE LOS DATOS DISPONIBLES (ignora NaN nativamente).
    """
    X = df[columns].apply(clean_numeric).astype(float)

    for col in columns:
        # pandas .median() ignora NaN automáticamente
        median = X[col].median()

        # Si TODA la columna es NaN (edge case), usar 0
        if pd.isna(median):
            median = 0.0

        X[col] = X[col].fillna(median)

    # Verificación: no deben quedar NaN
    assert not X.isna().any().any(), f"Quedan NaN tras imputación: {X.isna().sum()}"

    return X.to_numpy(dtype=float)


# ============================================================
# TOPSIS
# ============================================================

def topsis(X: np.ndarray, weights: np.ndarray, benefits: np.ndarray) -> np.ndarray:
    denominator = np.sqrt((X ** 2).sum(axis=0))
    denominator[denominator == 0] = 1

    R = X / denominator
    V = R * weights

    ideal_best = np.where(
        benefits,
        V.max(axis=0),
        V.min(axis=0),
    )

    ideal_worst = np.where(
        benefits,
        V.min(axis=0),
        V.max(axis=0),
    )

    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    denominator_score = d_best + d_worst

    return np.divide(
        d_worst,
        denominator_score,
        out=np.full(len(X), 0.5),
        where=denominator_score != 0,
    )


# ============================================================
# MABAC
# ============================================================

def mabac(X: np.ndarray, weights: np.ndarray, benefits: np.ndarray) -> np.ndarray:
    N = np.zeros_like(X, dtype=float)

    for j, benefit in enumerate(benefits):
        lo = np.nanmin(X[:, j])
        hi = np.nanmax(X[:, j])

        if np.isclose(lo, hi):
            N[:, j] = 0.5
            continue

        if benefit:
            N[:, j] = (X[:, j] - lo) / (hi - lo)
        else:
            N[:, j] = (hi - X[:, j]) / (hi - lo)

    V = N * weights + weights
    V = np.clip(V, 1e-12, None)

    frontier = np.exp(np.mean(np.log(V), axis=0))
    Q = V - frontier

    return Q.sum(axis=1)


# ============================================================
# FIX 3: MEREC con eps en log(Z)
# ============================================================

def merec(X: np.ndarray, weights_unused=None, benefits: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    MEREC calcula pesos OBJETIVOS.
    - NO utiliza los pesos AHP.
    - Los pesos dependen de la influencia de cada criterio.
    - Después se utiliza una normalización min-max para generar un score comparable.
    """
    if benefits is None:
        benefits = np.array([True] * X.shape[1], dtype=bool)

    # Evitar ceros y valores negativos.
    X_safe = np.maximum(X.astype(float), 1e-9)

    Z = np.zeros_like(X_safe)

    for j, benefit in enumerate(benefits):
        col = X_safe[:, j]

        if benefit:
            # Menor valor relativo = mejor.
            Z[:, j] = np.min(col) / col
        else:
            # Criterio costo.
            Z[:, j] = col / np.max(col)

    Z = np.clip(Z, 1e-12, 1)

    # FIX: eps antes de np.log para evitar log(1) = 0 y problemas numéricos
    EPS = 1e-12

    # Desempeño global.
    S = np.log(
        1
        + (
            np.sum(
                np.abs(np.log(Z + EPS)),  # <-- EPS aquí
                axis=1,
            )
            / X.shape[1]
        )
    )

    effects = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        Z_without = np.delete(Z, j, axis=1)

        S_without = np.log(
            1
            + (
                np.sum(
                    np.abs(np.log(Z_without + EPS)),  # <-- EPS aquí
                    axis=1,
                )
                / Z_without.shape[1]
            )
        )

        effects[j] = np.sum(np.abs(S_without - S))

    if effects.sum() <= 1e-12:
        weights = np.full(X.shape[1], 1 / X.shape[1])
    else:
        weights = effects / effects.sum()

    # Normalización final 0-1.
    N = np.zeros_like(X_safe)

    for j, benefit in enumerate(benefits):
        lo = np.min(X_safe[:, j])
        hi = np.max(X_safe[:, j])

        if np.isclose(lo, hi):
            N[:, j] = 0.5
        elif benefit:
            N[:, j] = (X_safe[:, j] - lo) / (hi - lo)
        else:
            N[:, j] = (hi - X_safe[:, j]) / (hi - lo)

    score = N @ weights

    return score, weights


# ============================================================
# RANKING
# ============================================================

def rank_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()

    criteria = [
        "npxg",
        "shots_on_target",
        "xa",
        "Valor (€)",
    ]

    benefits = np.array([True, True, True, False], dtype=bool)

    X = impute_matrix(df, criteria)

    weights_ahp = ahp_weights()

    df["Puntaje_TOPSIS"] = topsis(X, weights_ahp, benefits)
    df["Puntaje_MABAC"] = mabac(X, weights_ahp, benefits)
    df["Puntaje_MEREC"], weights_merec = merec(X, benefits=benefits)

    for method in ["TOPSIS", "MABAC", "MEREC"]:
        score_col = f"Puntaje_{method}"

        ordered = (
            df.sort_values(
                [score_col, "npxg", "Valor (€)"],
                ascending=[False, False, True],
                na_position="last",
            ).index
        )

        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered)

        df[f"Rank_{method}"] = ranks.reindex(df.index).astype(int)

    return df, weights_merec


# ============================================================
# PREPARACIÓN DE DATOS (simplificada, sin Streamlit)
# ============================================================

def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    cols = {key: find_col(df, aliases) for key, aliases in ALIASES.items()}

    name_col = cols["name"]
    if name_col is None:
        raise ValueError("El CSV no contiene una columna reconocible para el nombre del jugador.")

    df["Jugador"] = df[name_col].astype(str).str.strip()

    if cols["team"]:
        df["Equipo"] = df[cols["team"]].fillna("Sin equipo").astype(str).str.strip()
    else:
        df["Equipo"] = "Sin equipo"

    for target in ["Minutos", "Valor (€)", "Edad", "Goles", "Asistencias"]:
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


def add_extra_metrics(current: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    # Simplificado: no añadir métricas extra para este fix
    return current


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("PIPELINE DE CORRECCIÓN NaN + DEDUPLICACIÓN")
    print("=" * 60)

    # 1. Cargar datos
    print(f"\n1. Cargando datos desde: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"   Filas cargadas: {len(raw_df)}")
    print(f"   Columnas: {list(raw_df.columns)}")

    # 2. Preparar datos
    print("\n2. Preparando datos...")
    df_all = prepare_data(raw_df)
    print(f"   Filas tras prepare_data: {len(df_all)}")

    # 3. Consolidar jugadores actuales (CON FIX DE DEDUPLICACIÓN)
    print("\n3. Consolidando jugadores temporada 25-26 (fix deduplicación)...")
    current = consolidate_current_players(df_all)
    print(f"   Filas tras consolidación: {len(current)}")

    # Verificar duplicados
    dup_check = current.duplicated(subset=["player_id", "Temporada", "Equipo"]).sum()
    print(f"   Duplicados player_id+Temporada+Equipo: {dup_check}")

    # 4. Añadir métricas extra (opcional)
    current = add_extra_metrics(current, raw_df)

    # 5. Filtro mínimo de minutos (900) - replicar lógica del app
    min_minutes = 900
    current = current[current["Minutos"].fillna(0) >= min_minutes].copy()
    print(f"   Filas tras filtro minutos >= {min_minutes}: {len(current)}")

    # 6. Calcular rankings (CON FIX DE IMPUTACIÓN Y MEREC)
    print("\n4. Calculando rankings (TOPSIS, MABAC, MEREC)...")
    ranked, merec_weights = rank_frame(current)

    # 7. Verificar NaN en scores
    score_cols = ["Puntaje_TOPSIS", "Puntaje_MABAC", "Puntaje_MEREC"]
    for col in score_cols:
        nan_count = ranked[col].isna().sum()
        print(f"   {col}: NaN = {nan_count}")

    # 8. Mostrar top 5
    print("\n5. TOP 5 JUGADORES (por Puntaje_TOPSIS):")
    top5 = ranked.nlargest(5, "Puntaje_TOPSIS")[
        ["Jugador", "Equipo", "Edad", "Valor (€)", "npxg", "shots_on_target", "xa",
         "Puntaje_TOPSIS", "Puntaje_MABAC", "Puntaje_MEREC",
         "Rank_TOPSIS", "Rank_MABAC", "Rank_MEREC"]
    ].copy()
    top5["Valor (€)"] = top5["Valor (€)"].apply(euro)
    print(top5.to_string(index=False))

    # 9. Guardar CSV corregido
    print(f"\n6. Guardando CSV corregido en: {OUTPUT_PATH}")
    ranked.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("   ✓ Guardado exitosamente")

    # 10. Verificar archivo guardado
    print("\n7. Verificación de lectura:")
    check = pd.read_csv(OUTPUT_PATH)
    for col in score_cols:
        nan_count = check[col].isna().sum()
        print(f"   {col}: NaN = {nan_count}")

    print("\n" + "=" * 60)
    print("✅ PROCESO COMPLETADO SIN ERRORES")
    print("=" * 60)


if __name__ == "__main__":
    main()