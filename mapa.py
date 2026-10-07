"""Probability surface over UPZ (no Streamlit dependency, unit-testable).

Two modes:
  - historical: empirical share of incidents per UPZ over a period/type slice.
  - predicted : the supervised model's estimate per UPZ for a target month.

Both return the same schema (COD_UPZ, nombre_upz, localidad, total_incidentes,
prob_pct, indice_0_100) so the map code is identical for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_nuse


def _t(anio: int, mes: int) -> int:
    return int(anio) * 12 + int(mes)


def _indice_0_100(serie: pd.Series) -> pd.Series:
    mn, mx = float(serie.min()), float(serie.max())
    if mx == mn:
        return serie * 0.0
    return ((serie - mn) / (mx - mn) * 100).round(2)


def _base_upz() -> pd.DataFrame:
    return data_nuse.cargar_upz()[["COD_UPZ", "nombre_upz", "localidad"]].copy()


def _cerrar(base: pd.DataFrame, conteos: pd.DataFrame, col: str = "total_incidentes") -> pd.DataFrame:
    sup = base.merge(conteos, on="COD_UPZ", how="left")
    sup[col] = sup[col].fillna(0)
    total = float(sup[col].sum())
    sup["total_incidentes"] = sup[col].round().astype(int)
    sup["prob_pct"] = ((sup[col] / total * 100).round(4) if total > 0 else 0.0)
    sup["indice_0_100"] = _indice_0_100(sup[col])
    return sup.sort_values(col, ascending=False).reset_index(drop=True)


def superficie_historica(desde: tuple[int, int] | None = None,
                         hasta: tuple[int, int] | None = None,
                         tipo: str | None = None) -> pd.DataFrame:
    """Empirical share per UPZ over months in [desde, hasta] (inclusive) and type."""
    hechos = data_nuse.cargar_hechos().copy()
    hechos["_t"] = hechos["ANIO"] * 12 + hechos["MES"]
    if desde is not None:
        hechos = hechos[hechos["_t"] >= _t(*desde)]
    if hasta is not None:
        hechos = hechos[hechos["_t"] <= _t(*hasta)]
    if tipo is not None:
        hechos = hechos[hechos["TIPO_DETALLE"] == tipo]
    conteos = hechos.groupby("COD_UPZ", as_index=False)["total_incidentes"].sum()
    return _cerrar(_base_upz(), conteos)


def superficie_predicha(anio: int, mes: int) -> pd.DataFrame:
    """Model-predicted share per UPZ for a target month (uses history < month)."""
    import modelo

    pred = modelo.predecir_mes(anio, mes)
    return _cerrar(_base_upz(), pred, col="predicho")


def enriquecer_geojson(geo: dict, superficie: pd.DataFrame) -> dict:
    """Inject localidad / total / prob_pct / indice into each feature's properties."""
    valores = superficie.set_index("COD_UPZ").to_dict("index")
    for f in geo.get("features", []):
        cod = str(f["properties"].get("cod_upz", "")).strip().upper()
        f["properties"]["cod_upz"] = cod
        fila = valores.get(cod)
        if fila is None:
            f["properties"].update({"localidad": "", "total_incidentes": 0,
                                    "prob_pct": 0.0, "indice_0_100": 0.0})
        else:
            f["properties"].update({
                "localidad": str(fila["localidad"]),
                "total_incidentes": int(fila["total_incidentes"]),
                "prob_pct": float(fila["prob_pct"]),
                "indice_0_100": float(fila["indice_0_100"]),
            })
    return geo


def top_upz(superficie: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return superficie.head(n)[["nombre_upz", "localidad", "total_incidentes", "prob_pct"]]


_ESCALA_CACHE = None


def escala_prob_max() -> float:
    """Fixed colour-scale top: the maximum monthly UPZ share (%) in the history.

    Using a STABLE scale (instead of min-max per selection) makes the colour
    directly comparable to the probability value and keeps the map readable
    when the timeline moves.
    """
    global _ESCALA_CACHE
    if _ESCALA_CACHE is None:
        h = data_nuse.cargar_hechos()
        hm = h.groupby(["ANIO", "MES", "COD_UPZ"], as_index=False)["total_incidentes"].sum()
        total = hm.groupby(["ANIO", "MES"])["total_incidentes"].transform("sum").replace(0, np.nan)
        share = (hm["total_incidentes"] / total * 100)
        _ESCALA_CACHE = float(np.nanmax(share.to_numpy()))
    return _ESCALA_CACHE


def comparacion_mes(anio: int, mes: int) -> pd.DataFrame:
    """Predicted vs actual incidents per UPZ for a given month."""
    pred = (superficie_predicha(anio, mes)[["COD_UPZ", "nombre_upz", "localidad", "total_incidentes"]]
            .rename(columns={"total_incidentes": "predicho"}))
    hechos = data_nuse.cargar_hechos()
    real = (hechos[(hechos["ANIO"] == int(anio)) & (hechos["MES"] == int(mes))]
            .groupby("COD_UPZ", as_index=False)["total_incidentes"].sum()
            .rename(columns={"total_incidentes": "real"}))
    comp = pred.merge(real, on="COD_UPZ", how="left").fillna({"real": 0})
    comp["error"] = comp["predicho"] - comp["real"]
    return comp.sort_values("real", ascending=False).reset_index(drop=True)
