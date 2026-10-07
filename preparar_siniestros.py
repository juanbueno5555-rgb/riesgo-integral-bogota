"""Offline preparation of the georeferenced road-incident points (Siniestros Viales).

Source: Secretaría Distrital de Movilidad (SDM), Datos Abiertos Bogotá.
Reads the raw CSV and writes a compact points file used as a map layer:

    data/siniestros_puntos.csv.gz : lon, lat, anio, gravedad, clase, localidad

Usage:
    python preparar_siniestros.py [ruta_csv_crudo]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_CSV_DEFAULT = RAIZ_PROYECTO / "datos" / "siniestros.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"

LIMITES = {"lat_min": 4.4, "lat_max": 4.95, "lon_min": -74.30, "lon_max": -73.9}


def generar(ruta_csv=None) -> int:
    ruta = Path(ruta_csv) if ruta_csv else RUTA_CSV_DEFAULT
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el CSV de siniestros: {ruta}")

    df = pd.read_csv(
        ruta, dtype=str, encoding="utf-8-sig",
        usecols=["LATITUD", "LONGITUD", "ANO_OCURRENCIA_ACC", "GRAVEDAD", "CLASE_ACC", "LOCALIDAD"],
    )
    df = df.rename(columns={"LATITUD": "lat", "LONGITUD": "lon",
                            "ANO_OCURRENCIA_ACC": "anio", "GRAVEDAD": "gravedad",
                            "CLASE_ACC": "clase", "LOCALIDAD": "localidad"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(LIMITES["lat_min"], LIMITES["lat_max"])
            & df["lon"].between(LIMITES["lon_min"], LIMITES["lon_max"])]
    df["anio"] = df["anio"].fillna(0).astype(int)
    for c in ("gravedad", "clase", "localidad"):
        df[c] = df[c].fillna("").str.strip()
    df = df[["lon", "lat", "anio", "gravedad", "clase", "localidad"]].reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "siniestros_puntos.csv.gz", index=False, encoding="utf-8", compression="gzip")
    print(f"[siniestros] filas leidas: {n0:,} | puntos validos: {len(df):,} "
          f"| años {int(df['anio'].min())}-{int(df['anio'].max())}")
    print("[siniestros] escrito siniestros_puntos.csv.gz")
    return len(df)


def main() -> int:
    try:
        generar(sys.argv[1] if len(sys.argv) > 1 else None)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[siniestros] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
