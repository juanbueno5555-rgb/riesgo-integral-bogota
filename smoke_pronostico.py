"""Headless check for the forecast capability (months not published yet).

Complements smoke_check.py (which covers the historical pipeline) with the
recursive forecast path used by the app when the timeline is dragged past the
last published month. Read-only: no artifacts are written.

Run: python smoke_pronostico.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

import mapa  # noqa: E402
import modelo  # noqa: E402

OK = 0
FAIL = 0


def check(nombre: str, cond: bool, detalle: str = "") -> None:
    global OK, FAIL
    if cond:
        OK += 1
        print(f"OK  {nombre} {detalle}")
    else:
        FAIL += 1
        print(f"FAIL {nombre} {detalle}")


def main() -> int:
    meta_p = RAIZ / "data" / "pronostico_meta.json"
    check("artefacto de pronóstico", meta_p.exists(), meta_p.name)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    check("curva completa de horizontes",
          len(meta.get("error_por_horizonte", [])) == modelo.HORIZONTE_MAX)
    check("pronóstico completo de meses",
          len(meta.get("pronostico", [])) == modelo.HORIZONTE_MAX)

    t_ult = modelo.ultimo_t_publicado()
    check("último mes publicado derivado", modelo._etiqueta(t_ult) == meta.get("origen"),
          modelo._etiqueta(t_ult))
    check("horizonte del mes siguiente = 1", modelo.horizonte(*_siguiente(t_ult)) == 1)

    # El pronóstico recursivo debe coincidir con el artefacto que consume la app.
    esperado = {f["mes"]: f["total_predicho"] for f in meta.get("pronostico", [])}
    for etiqueta in list(esperado)[:1] + list(esperado)[-1:]:
        anio, mes = (int(x) for x in etiqueta.split("-"))
        df = modelo.pronosticar_futuro(anio, mes)
        total = int(round(df["predicho"].sum()))
        check(f"total forecast {etiqueta}", abs(total - esperado[etiqueta]) <= 2,
              f"{total:,} vs {esperado[etiqueta]:,}")
        check(f"112 UPZ {etiqueta}", len(df) == 112)

    # La superficie que pinta el mapa debe cerrar en 100 %.
    sup = mapa.superficie_pronosticada(*_siguiente(t_ult))
    check("superficie pronosticada 112 UPZ", len(sup) == 112)
    check("participación suma 100 %", abs(float(sup["prob_pct"].sum()) - 100.0) < 0.05,
          f"{sup['prob_pct'].sum():.3f}")

    # Más allá del horizonte evaluado se rechaza: el error no está medido.
    try:
        modelo.pronosticar_futuro(*_mas(t_ult, modelo.HORIZONTE_MAX + 1))
        check("rechaza horizonte fuera de rango", False)
    except ValueError:
        check("rechaza horizonte fuera de rango", True)

    print("=" * 48)
    print(f"SMOKE PRONOSTICO: {OK} OK, {FAIL} FAIL")
    return 1 if FAIL else 0


def _siguiente(t: int) -> tuple[int, int]:
    t += 1
    return (t - 1) // 12, (t - 1) % 12 + 1


def _mas(t: int, k: int) -> tuple[int, int]:
    t += k
    return (t - 1) // 12, (t - 1) % 12 + 1


if __name__ == "__main__":
    sys.exit(main())
