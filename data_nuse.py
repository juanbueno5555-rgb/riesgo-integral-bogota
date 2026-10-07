"""Lightweight access to the pre-aggregated NUSE data (monthly, by UPZ).

Only small derived files are loaded (never the raw CSV at runtime). If the
aggregated files are missing, it attempts to regenerate them from the raw CSV
through preparar_datos.py. No Streamlit dependency here (headless-testable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

ARCHIVOS = {
    "upz": "upz_probabilidad.csv",
    "hechos": "upz_anio_mes_tipo.csv.gz",
    "anio": "nuse_por_anio.csv",
    "mes": "nuse_por_mes.csv",
    "geo": "upz_geo.geojson",
    "proveniencia": "proveniencia.json",
}


def datos_disponibles() -> bool:
    return all((DATA_DIR / nombre).exists() for nombre in ARCHIVOS.values())


def generar_desde_crudo() -> None:
    import preparar_datos

    preparar_datos.generar()


def _asegurar_datos() -> None:
    if not datos_disponibles():
        print("[data_nuse] Datos agregados ausentes; generando desde el CSV crudo...")
        generar_desde_crudo()
    if not datos_disponibles():
        raise FileNotFoundError("Faltan los datos agregados. Ejecute: python preparar_datos.py")


def cargar_upz() -> pd.DataFrame:
    """Incidents per UPZ over the whole period (+ share % and 0-100 index)."""
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["upz"], encoding="utf-8-sig")


def cargar_hechos() -> pd.DataFrame:
    """Monthly fact table: UPZ x year x month x detail type (gzip CSV)."""
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["hechos"], encoding="utf-8-sig")


def cargar_por_anio() -> pd.DataFrame:
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["anio"], encoding="utf-8-sig")


def cargar_por_mes() -> pd.DataFrame:
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["mes"], encoding="utf-8-sig")


def cargar_geojson() -> dict:
    _asegurar_datos()
    with open(DATA_DIR / ARCHIVOS["geo"], encoding="utf-8") as fh:
        return json.load(fh)


def cargar_proveniencia() -> dict:
    _asegurar_datos()
    with open(DATA_DIR / ARCHIVOS["proveniencia"], encoding="utf-8") as fh:
        return json.load(fh)


def cargar_localidad_geo() -> dict:
    """Localidad boundaries (optional layer); empty FeatureCollection if absent."""
    _asegurar_datos()
    ruta = DATA_DIR / "localidad_geo.geojson"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {"type": "FeatureCollection", "features": []}


def cargar_siniestros() -> pd.DataFrame:
    """Georeferenced road-incident points (optional layer); empty if absent."""
    _asegurar_datos()
    ruta = DATA_DIR / "siniestros_puntos.csv.gz"
    if ruta.exists():
        return pd.read_csv(ruta)
    return pd.DataFrame(columns=["lon", "lat", "anio", "gravedad", "clase", "localidad"])


def listar_tipos(top: int | None = None) -> list[str]:
    hechos = cargar_hechos()
    totales = hechos.groupby("TIPO_DETALLE")["total_incidentes"].sum().sort_values(ascending=False)
    tipos = totales.index.tolist()
    return tipos if top is None else tipos[:top]


def periodos() -> list[tuple[int, int]]:
    """Sorted list of (year, month) present in the monthly fact table."""
    hechos = cargar_hechos()
    pares = hechos[["ANIO", "MES"]].drop_duplicates().sort_values(["ANIO", "MES"])
    return [(int(a), int(m)) for a, m in pares.to_numpy()]


def etiqueta(anio: int, mes: int) -> str:
    return f"{int(anio)}-{int(mes):02d}"
