"""Smoke check for the RIESGO INTEGRAL MVP (no Streamlit required).

Verifies, in order:
  1. aggregated NUSE data are present and sane
  2. synthetic sample points are present and inside Bogota bounds
  3. the demo model trains/loads and reports AUC on holdout
  4. predecir() returns a valid probability and 0-100 exposure
  5. hazard_score / exposure_score / score_integral return 0-100 values
  6. provenance note is readable

Prints an OK/FAIL line per check and exits 0 only if every check passes.

Usage:
    python smoke_check.py
"""

from __future__ import annotations

import sys

import data_nuse
import integracion
import modelo_riesgo


def main() -> int:
    print("=== SMOKE CHECK: RIESGO INTEGRAL MVP ===")
    fallos = 0
    aciertos = 0

    def ok():
        nonlocal aciertos
        aciertos += 1

    def fail():
        nonlocal fallos
        fallos += 1

    # 1. Aggregated NUSE data ------------------------------------------------
    try:
        resumen = data_nuse.cargar_resumen()
        assert len(resumen) >= 15, "menos de 15 localidades válidas"
        assert {"COD_LOCALIDAD", "LOCALIDAD", "total_incidentes", "peligro_0_100",
                "proporcion"}.issubset(resumen.columns)
        p_ok = resumen["peligro_0_100"].between(0, 100).all()
        prop_ok = abs(resumen["proporcion"].sum() - 1.0) < 0.01
        assert p_ok and prop_ok
        print(f"OK  datos agregados NUSE: {len(resumen)} localidades, "
              f"total {int(resumen['total_incidentes'].sum()):,} incidentes")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL datos agregados NUSE: {exc}")
        fail()

    # 2. Sample points --------------------------------------------------------
    try:
        puntos = data_nuse.cargar_puntos()
        n = len(puntos)
        assert 4000 <= n <= 6000, f"tamaño de muestra inesperado: {n}"
        lat_ok = puntos["lat"].between(4.45, 4.90).all()
        lon_ok = puntos["lon"].between(-74.25, -73.95).all()
        assert lat_ok and lon_ok, "puntos fuera de los límites de Bogotá"
        assert "clase" in puntos.columns
        print(f"OK  puntos de muestra: {n} puntos sintéticos dentro de Bogotá "
              f"({int(puntos['clase'].ne('').sum())} marcados)")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL puntos de muestra: {exc}")
        fail()

    # 3. Modelo demo (train + AUC) -------------------------------------------
    try:
        metricas = modelo_riesgo.entrenar()
        auc = metricas["auc"]
        assert auc >= 0.60, f"AUC demasiado baja para superficie útil: {auc}"
        print(f"OK  modelo demo: AUC (holdout) = {auc:.4f}, "
              f"prevalencia = {metricas['prevalencia']:.1%}, "
              f"n = {metricas['n_filas']}")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL modelo demo: {exc}")
        fail()

    # 4. Predicción de un perfil ---------------------------------------------
    try:
        perfil = {
            "edad": 55, "bmi": 29.0, "hipertension": 0, "colesterol_alto": 1,
            "actividad_fisica": 0, "fumador": 0, "salud_general": 3,
        }
        resultado = modelo_riesgo.predecir(perfil)
        prob = resultado["probabilidad"]
        exp = resultado["exposicion_0_100"]
        assert 0.0 < prob < 1.0, f"probabilidad fuera de rango: {prob}"
        assert 0.0 <= exp <= 100.0, f"exposición fuera de rango: {exp}"
        assert len(resultado["contribuciones"]) == len(modelo_riesgo.CARACTERISTICAS)
        print(f"OK  predecir(perfil): probabilidad = {prob:.1%}, "
              f"exposición 0-100 = {exp:.2f}, {len(resultado['contribuciones'])} factores")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL predecir(perfil): {exc}")
        fail()

    # 5. Scores de integración ------------------------------------------------
    try:
        haz = integracion.hazard_score(resumen.iloc[0]["LOCALIDAD"])
        assert 0.0 <= haz["peligro_0_100"] <= 100.0
        exp2 = integracion.exposure_score(0.25)
        assert abs(exp2 - 25.0) < 0.01
        integral = integracion.score_integral(haz["peligro_0_100"], exp2)
        esperado = (haz["peligro_0_100"] * exp2) / 100.0
        assert abs(integral["integral_0_100"] - esperado) < 0.2
        assert integral["banda"] in ("Riesgo bajo", "Riesgo medio", "Riesgo alto")
        print(f"OK  score integral: peligro {haz['peligro_0_100']:.1f} x "
              f"exposición {exp2:.1f} = integral {integral['integral_0_100']:.1f} "
              f"({integral['banda']})")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL scores de integración: {exc}")
        fail()

    # 6. Provenance -----------------------------------------------------------
    try:
        prov = data_nuse.cargar_proveniencia()
        periodo = prov["periodo"]
        assert periodo["anio_min"] and periodo["anio_max"]
        print(f"OK  proveniencia: fuente={prov['fuente'].split(chr(92))[-1]}, "
              f"periodo {periodo['anio_min']}-{periodo['anio_max']}, "
              f"excluidos={prov['reglas_filtro']['excluidos']}")
        ok()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL proveniencia: {exc}")
        fail()

    print("=" * 48)
    print(f"SMOKE CHECK: {aciertos} OK, {fallos} FAIL")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
