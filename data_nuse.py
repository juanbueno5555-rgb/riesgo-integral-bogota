"""Lightweight access to the pre-aggregated NUSE data for the MVP.

Only small derived files are loaded (no full 113 MB CSV at runtime). If the
aggregated files are missing, this module attempts to regenerate them from the
raw CSV through preparar_datos.py. No Streamlit dependency here so that the
data layer can be exercised headlessly (smoke_check.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

ARCHIVOS = {
    "resumen": "nuse_por_localidad.csv",
    "por_anio": "nuse_por_anio.csv",
    "top_tipos": "nuse_top_tipos.csv",
    "puntos": "puntos_muestra.csv",
    "proveniencia": "proveniencia.json",
}


def datos_disponibles() -> bool:
    return all((DATA_DIR / nombre).exists() for nombre in ARCHIVOS.values())


def generar_desde_crudo() -> None:
    """Regenerate aggregates from the raw CSV (slow; intended as fallback)."""
    import preparar_datos

    preparar_datos.generar()


def _asegurar_datos() -> None:
    if not datos_disponibles():
        print("[data_nuse] Datos agregados ausentes; generando desde el CSV crudo...")
        generar_desde_crudo()
    if not datos_disponibles():
        raise FileNotFoundError(
            "No se encontraron los datos agregados. Ejecute: python preparar_datos.py"
        )


def cargar_resumen() -> pd.DataFrame:
    """Incidents per localidad with 0-100 normalized hazard score."""
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["resumen"], encoding="utf-8-sig")


def cargar_por_anio() -> pd.DataFrame:
    """Incidents per localidad per year (long format)."""
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["por_anio"], encoding="utf-8-sig")


def cargar_top_tipos(n: int = 15) -> pd.DataFrame:
    """Top incident detail types by count."""
    _asegurar_datos()
    df = pd.read_csv(DATA_DIR / ARCHIVOS["top_tipos"], encoding="utf-8-sig")
    return df.head(n)


def cargar_puntos() -> pd.DataFrame:
    """Synthetic sample points (heatmap) weighted by the real distribution."""
    _asegurar_datos()
    return pd.read_csv(DATA_DIR / ARCHIVOS["puntos"], encoding="utf-8-sig")


def cargar_proveniencia() -> dict:
    _asegurar_datos()
    import json

    with open(DATA_DIR / ARCHIVOS["proveniencia"], encoding="utf-8") as fh:
        return json.load(fh)


def listar_localidades() -> list[str]:
    """Ordered list of localidad names present in the aggregates."""
    resumen = cargar_resumen()
    return resumen["LOCALIDAD"].tolist()


def periodo() -> tuple[int, int] | None:
    """(min_year, max_year) from the provenance note, if available."""
    try:
        prov = cargar_proveniencia()
        return prov["periodo"]["anio_min"], prov["periodo"]["anio_max"]
    except (KeyError, OSError, ValueError):
        return None
