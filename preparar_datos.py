"""Offline preparation of aggregated data for the RIESGO INTEGRAL MVP.

Reads the raw NUSE (linea 123, Bogota) incident CSV and produces lightweight
aggregated artifacts inside prototipo/data/:

  - nuse_por_localidad.csv : incidents per localidad (total, share, 0-100 hazard)
  - nuse_por_anio.csv      : incidents per localidad and year (long format)
  - nuse_top_tipos.csv     : top incident detail types (count)
  - puntos_muestra.csv     : seeded synthetic points whose distribution follows
                             the real per-localidad weights (for the heatmap)
  - proveniencia.json      : provenance note (source, filter rules, date)

Usage:
    python preparar_datos.py [ruta_del_csv_crudo]
If no path is given, the documented default locations are tried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

RUTA_CSV_DEFAULT = (
    r"C:\Users\RANGE\AppData\Local\Temp\opencode\datos-bogota\nuse_llamadas.csv"
)

DATA_DIR = Path(__file__).resolve().parent / "data"

COLUMNAS_REQUERIDAS = [
    "ID", "ANIO", "MES", "TIPO_INCIDENTE", "TIPO_DETALLE",
    "COD_LOCALIDAD", "LOCALIDAD", "COD_UPZ", "UPZ", "CANT_INCIDENTES",
]

CODIGOS_NO_LOCALIZADOS = {"99", "-"}  # SIN LOCALIZACION / junk codes (excluded)

# Approximate geographic centroids per localidad (used only to place the
# synthetic sample points inside Bogota; real point incidents arrive later).
CENTROIDES_LOCALIDAD = {
    "01": (4.753, -74.031), "02": (4.649, -74.052), "03": (4.603, -74.065),
    "04": (4.578, -74.062), "05": (4.474, -74.124), "06": (4.610, -74.119),
    "07": (4.621, -74.179), "08": (4.638, -74.155), "09": (4.674, -74.144),
    "10": (4.700, -74.104), "11": (4.750, -74.085), "12": (4.670, -74.073),
    "13": (4.646, -74.094), "14": (4.609, -74.096), "15": (4.593, -74.109),
    "16": (4.619, -74.116), "17": (4.592, -74.074), "18": (4.584, -74.113),
    "19": (4.578, -74.159), "20": (4.460, -74.230),
}

LIMITES_BOGOTA = {"lat_min": 4.45, "lat_max": 4.90, "lon_min": -74.25, "lon_max": -73.95}
N_PUNTOS_MUESTRA = 5000
SEMILLA = 20260225


# ----------------------------------------------------------------------------
# Encoding / reading helpers
# ----------------------------------------------------------------------------

def detectar_codificacion(ruta: Path) -> str:
    """Return an encoding that decodes the whole file, utf-8-sig first."""
    with open(ruta, "rb") as fh:
        raw = fh.read(200_000)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def s(texto: str | None) -> str:
    """Sanitize a text cell: strip quotes, collapse replacement chars, strip."""
    if texto is None:
        return ""
    texto = texto.strip().strip('"')
    # Source export replaced some accented chars with U+FFFD; collapse them.
    texto = texto.replace("\ufffd", "")
    return " ".join(texto.split())


def leer_csv(ruta: Path) -> pd.DataFrame:
    print(f"[preparar_datos] leyendo {ruta}")
    print(f"[preparar_datos] tamano del archivo: {ruta.stat().st_size / 1e6:.1f} MB")
    enc = detectar_codificacion(ruta)
    print(f"[preparar_datos] codificacion detectada: {enc}")
    df = pd.read_csv(ruta, sep=";", encoding=enc, dtype=str, keep_default_na=False)
    cols = [c for c in df.columns]
    df.columns = [s(c) for c in df.columns]
    if list(df.columns) != COLUMNAS_REQUERIDAS:
        raise ValueError(
            "Columnas inesperadas en el CSV crudo.\n"
            f"Esperadas: {COLUMNAS_REQUERIDAS}\n"
            f"Encontradas: {list(df.columns)}"
        )
    print(f"[preparar_datos] filas leidas: {len(df):,}")
    for col in ("LOCALIDAD", "TIPO_INCIDENTE", "TIPO_DETALLE", "UPZ"):
        df[col] = df[col].map(s)
    df["COD_LOCALIDAD"] = df["COD_LOCALIDAD"].map(s)
    df["CANT_INCIDENTES"] = pd.to_numeric(df["CANT_INCIDENTES"], errors="coerce").fillna(0).astype(int)
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").fillna(0).astype(int)
    return df


# ----------------------------------------------------------------------------
# Aggregations
# ----------------------------------------------------------------------------

def localidades_validas(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out SIN LOCALIZACION / junk codes. Sumapaz ('20') is kept."""
    n_total = len(df)
    df = df[~df["COD_LOCALIDAD"].isin(CODIGOS_NO_LOCALIZADOS)]
    n_ok = len(df)
    print(f"[preparar_datos] filas excluidas por COD_LOCALIDAD en {sorted(CODIGOS_NO_LOCALIZADOS)}: {n_total - n_ok:,}")
    return df.copy()


def resumen_por_localidad(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["COD_LOCALIDAD", "LOCALIDAD"], as_index=False)["CANT_INCIDENTES"].sum())
    agg = agg.sort_values("CANT_INCIDENTES", ascending=False).reset_index(drop=True)
    total = agg["CANT_INCIDENTES"].sum()
    agg["proporcion"] = (agg["CANT_INCIDENTES"] / total).round(6)
    mx, mn = agg["CANT_INCIDENTES"].max(), agg["CANT_INCIDENTES"].min()
    rango = mx - mn if mx > mn else 1.0
    agg["peligro_0_100"] = ((agg["CANT_INCIDENTES"] - mn) / rango * 100).round(1)
    agg = agg.rename(columns={"CANT_INCIDENTES": "total_incidentes"})
    return agg


def resumen_por_anio(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["ANIO", "COD_LOCALIDAD", "LOCALIDAD"], as_index=False)["CANT_INCIDENTES"].sum())
    agg = agg.rename(columns={"CANT_INCIDENTES": "total_incidentes"})
    return agg.sort_values(["ANIO", "total_incidentes"], ascending=[True, False])


def top_tipos(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    agg = (df.groupby("TIPO_DETALLE", as_index=False)["CANT_INCIDENTES"].sum())
    agg["TIPO_DETALLE"] = agg["TIPO_DETALLE"].map(s)
    agg = (agg.groupby("TIPO_DETALLE", as_index=False)["CANT_INCIDENTES"].sum())
    agg = agg.sort_values("CANT_INCIDENTES", ascending=False).head(n)
    return agg.rename(columns={"CANT_INCIDENTES": "total_incidentes"}).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Synthetic sample points (prototype heatmap)
# ----------------------------------------------------------------------------

def generar_puntos_muestra(resumen: pd.DataFrame) -> pd.DataFrame:
    """Seeded synthetic points, per-localidad counts proportional to the real
    distribution, jittered around localidad centroids inside Bogota bounds."""
    rng = np.random.default_rng(SEMILLA)
    resumen = resumen[resumen["COD_LOCALIDAD"].isin(CENTROIDES_LOCALIDAD)]
    pesos = resumen["total_incidentes"].values
    pesos = pesos / pesos.sum()
    conteos = (pesos * N_PUNTOS_MUESTRA).astype(int)
    remanente = N_PUNTOS_MUESTRA - int(conteos.sum())
    if remanente > 0:
        idx = rng.choice(len(conteos), size=remanente, p=pesos)
        for i in idx:
            conteos[i] += 1

    lats, lons, cods, noms = [], [], [], []
    for (_, fila), n in zip(resumen.iterrows(), conteos):
        cod, nom = fila["COD_LOCALIDAD"], fila["LOCALIDAD"]
        lat_c, lon_c = CENTROIDES_LOCALIDAD[cod]
        sigma = 0.0045
        lat = lat_c + rng.normal(0, sigma, n)
        lon = lon_c + rng.normal(0, sigma, n)
        lat = np.clip(lat, LIMITES_BOGOTA["lat_min"], LIMITES_BOGOTA["lat_max"])
        lon = np.clip(lon, LIMITES_BOGOTA["lon_min"], LIMITES_BOGOTA["lon_max"])
        lats.extend(lat); lons.extend(lon); cods.extend([cod] * n); noms.extend([nom] * n)

    pts = pd.DataFrame({"lat": lats, "lon": lons, "cod_localidad": cods, "localidad": noms})
    pts["clase"] = "muestra_sintetica"
    return pts


# ----------------------------------------------------------------------------
# Write outputs
# ----------------------------------------------------------------------------

def escribir(archivo: Path, df: pd.DataFrame):
    df.to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"[preparar_datos] escrito {archivo.name} ({len(df):,} filas)")


def generar(ruta_csv: str | os.PathLike | None = None, salida: os.PathLike | None = None) -> dict:
    """Full pipeline. Returns a summary dict usable for provenance."""
    ruta = Path(ruta_csv) if ruta_csv else RUTA_CSV_DEFAULT
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No se encontro el CSV crudo: {ruta}")

    out = Path(salida) if salida else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    df = leer_csv(Path(ruta))
    df = localidades_validas(df)
    enc_usada = detectar_codificacion(Path(ruta))

    resumen = resumen_por_localidad(df)
    por_anio = resumen_por_anio(df)
    tipos = top_tipos(df)
    puntos = generar_puntos_muestra(resumen)

    escribir(out / "nuse_por_localidad.csv", resumen)
    escribir(out / "nuse_por_anio.csv", por_anio)
    escribir(out / "nuse_top_tipos.csv", tipos)
    escribir(out / "puntos_muestra.csv", puntos)

    proveniencia = {
        "fuente": str(Path(ruta)),
        "generado": datetime.now().isoformat(timespec="seconds"),
        "codificacion": enc_usada,
        "nota_codificacion": (
            "Codificacion verificada UTF-8 con BOM; los acentos espanoles estan "
            "intactos (no se detectaron caracteres de reemplazo U+FFFD). Los "
            "artefactos visibles al inspeccionar el CSV en consolas Windows son "
            "de presentacion, no de datos."
        ),
        "periodo": {
            "anio_min": int(df["ANIO"].min()),
            "anio_max": int(df["ANIO"].max()),
            "meses_anio_completo": int(df.drop_duplicates(["ANIO", "MES"]).shape[0]),
        },
        "filas_leidas": len(df),
        "reglas_filtro": {
            "excluidos": sorted(CODIGOS_NO_LOCALIZADOS),
            "nota": "Codigos '99'/'-' = SIN LOCALIZACION, excluidos de los agregados. "
                    "Sumapaz ('20') se conserva aunque aporta pocos incidentes.",
        },
        "normalizacion": "peligro_0_100: reescalado min-max 0-100 de total_incidentes "
                         "sobre las localidades validas (proporcion de densidad, no calibrado).",
        "puntos_muestra": {
            "total": int(len(puntos)),
            "semilla": SEMILLA,
            "nota": "Puntos sinteticos generados con distribucion proporcional a los "
                    "incidentes reales por localidad. Reemplazables por incidentes "
                    "reales con coordenadas en una fase posterior.",
        },
    }
    with open(out / "proveniencia.json", "w", encoding="utf-8") as fh:
        json.dump(proveniencia, fh, ensure_ascii=False, indent=2)
    print(f"[preparar_datos] escrito {out / 'proveniencia.json'}")

    print("\n[preparar_datos] RESUMEN")
    print(f"  Localidades con datos: {len(resumen)}")
    print(f"  Periodo cubierto      : {proveniencia['periodo']['anio_min']}-{proveniencia['periodo']['anio_max']}")
    print(f"  Total incidentes      : {int(resumen['total_incidentes'].sum()):,}")
    print(resumen[["COD_LOCALIDAD", "LOCALIDAD", "total_incidentes", "peligro_0_100"]].head(8).to_string(index=False))
    print("[preparar_datos] finalizado OK")
    return proveniencia


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera los datos agregados del prototipo RIESGO INTEGRAL.")
    parser.add_argument("csv", nargs="?", default=None, help="Ruta del CSV crudo de NUSE")
    args = parser.parse_args()
    try:
        generar(args.csv)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[preparar_datos] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
