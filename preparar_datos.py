"""Offline preparation of aggregated data for the Sentinel MVP.

Focus: probability map of emergency incidents (NUSE, Linea 123, Bogota),
aggregated by UPZ (Unidad de Planeamiento Zonal).

Reads the raw NUSE CSV and the UPZ GeoJSON, and produces lightweight
artifacts inside data/:

  - upz_probabilidad.csv : incidents per UPZ (total, share %, 0-100 index)
  - upz_por_anio.csv     : incidents per UPZ per year (+ within-year share)
  - upz_por_tipo.csv     : incidents per UPZ per detail type
  - nuse_por_anio.csv    : total incidents per year (trend)
  - upz_geo.geojson      : UPZ polygons (deduped, UPZ only)
  - proveniencia.json    : provenance note (source, filters, date)

Usage:
    python preparar_datos.py [ruta_csv_crudo] [ruta_geojson]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]  # Programacion-IA.Proyecto/
RUTA_CSV_DEFAULT = RAIZ_PROYECTO / "datos" / "nuse_llamadas.csv"
RUTA_GEOJSON_DEFAULT = RAIZ_PROYECTO / "datos" / "upz.geojson"
RUTA_LOCALIDAD_DEFAULT = RAIZ_PROYECTO / "datos" / "localidad.geojson"
DATA_DIR = Path(__file__).resolve().parent / "data"

COLUMNAS_REQUERIDAS = [
    "ID", "ANIO", "MES", "TIPO_INCIDENTE", "TIPO_DETALLE",
    "COD_LOCALIDAD", "LOCALIDAD", "COD_UPZ", "UPZ", "CANT_INCIDENTES",
]

CODIGOS_NO_LOCALIZADOS_LOC = {"99", "-"}
CODIGO_UPZ_SIN_LOC = "UPZ999"


def _norm(texto: str | None) -> str:
    if texto is None:
        return ""
    texto = str(texto).strip().strip('"').replace("\ufffd", "")
    return " ".join(texto.split())


def leer_geojson_codigos(ruta: Path) -> set[str]:
    """UPZ codes present in the GeoJSON (deduped, UPZ only)."""
    gj = json.loads(ruta.read_text(encoding="utf-8"))
    codigos = set()
    for f in gj.get("features", []):
        cod = _norm(f["properties"].get("COD_UPZ")).upper()
        if cod.startswith("UPZ"):
            codigos.add(cod)
    return codigos


def limpiar_geojson(ruta: Path, destino: Path) -> int:
    """Write a deduped UPZ-only GeoJSON, reprojected to WGS84 lon/lat.

    The source GeoJSON is in EPSG:3857 (Web Mercator, meters); Folium/Leaflet
    needs EPSG:4326 (lon, lat).
    """
    R = 6378137.0

    def a_lonlat(easting: float, northing: float) -> list[float]:
        lon = math.degrees(easting / R)
        lat = math.degrees(2 * math.atan(math.exp(northing / R)) - math.pi / 2)
        return [round(lon, 6), round(lat, 6)]

    def reproyectar_ring(ring):
        return [a_lonlat(x, y) for x, y in ring]

    def reproyectar_geom(g):
        if g["type"] == "Polygon":
            return {"type": "Polygon", "coordinates": [reproyectar_ring(r) for r in g["coordinates"]]}
        if g["type"] == "MultiPolygon":
            return {"type": "MultiPolygon",
                    "coordinates": [[reproyectar_ring(r) for r in poly] for poly in g["coordinates"]]}
        return g

    gj = json.loads(ruta.read_text(encoding="utf-8"))
    vistos, features = set(), []
    for f in gj.get("features", []):
        p = f["properties"]
        cod = _norm(p.get("COD_UPZ")).upper()
        if not cod.startswith("UPZ") or cod in vistos:
            continue
        vistos.add(cod)
        features.append({
            "type": "Feature",
            "properties": {"cod_upz": cod, "nombre_upz": _norm(p.get("NOMBRE_UPZ"))},
            "geometry": reproyectar_geom(f["geometry"]),
        })
    destino.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(features)


def preparar_localidades(ruta: Path, destino: Path) -> int:
    """Convert the Esri-JSON localidad boundaries (already lon/lat) to GeoJSON."""
    data = json.loads(ruta.read_text(encoding="utf-8"))
    features = []
    for f in data.get("features", []):
        attrs = f.get("attributes", {})
        rings = (f.get("geometry") or {}).get("rings")
        if not rings:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "localidad": _norm(attrs.get("LocNombre")),
                "cod_localidad": str(attrs.get("LocCodigo", "")).strip(),
            },
            "geometry": {"type": "MultiPolygon",
                         "coordinates": [[[[x, y] for x, y in ring]] for ring in rings]},
        })
    destino.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(features)


def leer_csv(ruta: Path) -> pd.DataFrame:
    print(f"[preparar_datos] leyendo {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [_norm(c) for c in df.columns]
    if list(df.columns) != COLUMNAS_REQUERIDAS:
        raise ValueError(f"Columnas inesperadas: {list(df.columns)}")
    for col in ("LOCALIDAD", "TIPO_DETALLE", "UPZ"):
        df[col] = df[col].map(_norm)
    df["COD_UPZ"] = df["COD_UPZ"].map(lambda x: _norm(x).upper())
    df["COD_LOCALIDAD"] = df["COD_LOCALIDAD"].map(lambda x: _norm(x).upper())
    df["CANT_INCIDENTES"] = pd.to_numeric(df["CANT_INCIDENTES"], errors="coerce").fillna(0).astype(int)
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    print(f"[preparar_datos] filas leidas: {len(df):,}")
    return df


def filtrar(df: pd.DataFrame, codigos_upz: set[str]) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df["COD_LOCALIDAD"].isin(CODIGOS_NO_LOCALIZADOS_LOC)]
    df = df[df["COD_UPZ"].isin(codigos_upz)]
    print(f"[preparar_datos] filas excluidas (sin localizacion / UPZ no valida): {n0 - len(df):,}")
    return df.copy()


def indice_0_100(serie: pd.Series) -> pd.Series:
    mn, mx = serie.min(), serie.max()
    if mx == mn:
        return serie * 0.0
    return ((serie - mn) / (mx - mn) * 100).round(2)


def generar(ruta_csv=None, ruta_geojson=None) -> dict:
    ruta_csv = Path(ruta_csv) if ruta_csv else RUTA_CSV_DEFAULT
    ruta_geojson = Path(ruta_geojson) if ruta_geojson else RUTA_GEOJSON_DEFAULT
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontro el CSV crudo: {ruta_csv}")
    if not ruta_geojson.exists():
        raise FileNotFoundError(f"No se encontro el GeoJSON de UPZ: {ruta_geojson}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    codigos_upz = leer_geojson_codigos(ruta_geojson)
    print(f"[preparar_datos] UPZ en el GeoJSON: {len(codigos_upz)}")

    df = filtrar(leer_csv(ruta_csv), codigos_upz)

    total = int(df["CANT_INCIDENTES"].sum())

    # Por UPZ (total del periodo). Se agrupa solo por codigo: algunos codigos
    # traen variantes de nombre/localidad en el crudo (se toma la mas frecuente).
    def _moda(serie: pd.Series) -> str:
        m = serie.mode()
        return m.iat[0] if len(m) else (serie.iloc[0] if len(serie) else "")

    upz = (df.groupby("COD_UPZ", as_index=False)
             .agg(total_incidentes=("CANT_INCIDENTES", "sum"),
                  nombre_upz=("UPZ", _moda),
                  localidad=("LOCALIDAD", _moda)))
    upz["prob_pct"] = (upz["total_incidentes"] / total * 100).round(4)
    upz["indice_0_100"] = indice_0_100(upz["total_incidentes"])
    upz = upz.sort_values("total_incidentes", ascending=False).reset_index(drop=True)

    # Tabla de hechos MENSUAL: UPZ x anio x mes x tipo (timeline + filtro de tipo)
    hechos = (df.groupby(["COD_UPZ", "ANIO", "MES", "TIPO_DETALLE"], as_index=False)["CANT_INCIDENTES"].sum()
                .rename(columns={"CANT_INCIDENTES": "total_incidentes"}))
    hechos["MES"] = hechos["MES"].astype(int)

    # Totales por anio y por mes (tendencia)
    por_anio_total = (df.groupby("ANIO", as_index=False)["CANT_INCIDENTES"].sum()
                        .rename(columns={"CANT_INCIDENTES": "total_incidentes"})
                        .sort_values("ANIO"))
    por_anio_total["prob_pct"] = (por_anio_total["total_incidentes"] / total * 100).round(4)
    por_mes_total = (df.groupby(["ANIO", "MES"], as_index=False)["CANT_INCIDENTES"].sum()
                       .rename(columns={"CANT_INCIDENTES": "total_incidentes"})
                       .sort_values(["ANIO", "MES"]))

    upz.to_csv(DATA_DIR / "upz_probabilidad.csv", index=False, encoding="utf-8-sig")
    hechos.to_csv(DATA_DIR / "upz_anio_mes_tipo.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    por_anio_total.to_csv(DATA_DIR / "nuse_por_anio.csv", index=False, encoding="utf-8-sig")
    por_mes_total.to_csv(DATA_DIR / "nuse_por_mes.csv", index=False, encoding="utf-8-sig")
    n_geo = limpiar_geojson(ruta_geojson, DATA_DIR / "upz_geo.geojson")
    n_loc = 0
    if RUTA_LOCALIDAD_DEFAULT.exists():
        n_loc = preparar_localidades(RUTA_LOCALIDAD_DEFAULT, DATA_DIR / "localidad_geo.geojson")

    proveniencia = {
        "fuente_datos": str(ruta_csv),
        "fuente_geo": str(ruta_geojson),
        "generado": datetime.now().isoformat(timespec="seconds"),
        "periodo": {"anio_min": int(df["ANIO"].min()), "anio_max": int(df["ANIO"].max())},
        "filas_leidas": int(len(df)),
        "total_incidentes": total,
        "upz_con_datos": int(len(upz)),
        "upz_en_geojson": n_geo,
        "reglas_filtro": {
            "excluidos_localidad": sorted(CODIGOS_NO_LOCALIZADOS_LOC),
            "excluidos_upz": [CODIGO_UPZ_SIN_LOC, "UPZ990-996", "UPR*"],
            "nota": "Se conservan solo UPZ presentes en el GeoJSON oficial. Sin localizacion/UPZ especiales se excluyen.",
        },
        "probabilidad": "prob_pct = participacion de la UPZ en el total del periodo (probabilidad empirica de incidencia). indice_0_100 = min-max para color.",
    }
    (DATA_DIR / "proveniencia.json").write_text(
        json.dumps(proveniencia, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[preparar_datos] RESUMEN")
    print(f"  Periodo        : {proveniencia['periodo']['anio_min']}-{proveniencia['periodo']['anio_max']}")
    print(f"  Total incidentes: {total:,}")
    print(f"  UPZ con datos  : {len(upz)}")
    print(upz[["COD_UPZ", "nombre_upz", "localidad", "total_incidentes", "prob_pct"]].head(8).to_string(index=False))
    print("[preparar_datos] finalizado OK")
    return proveniencia


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera los agregados por UPZ del MVP Sentinel.")
    ap.add_argument("csv", nargs="?", default=None)
    ap.add_argument("geojson", nargs="?", default=None)
    args = ap.parse_args()
    try:
        generar(args.csv, args.geojson)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[preparar_datos] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
