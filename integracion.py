"""Integration of territorial hazard (Module A) and personal exposure
(Module B) into an integral risk score, ISO 31000 framing (Risk = Hazard x
Exposure).

Scoring design (MVP):
  - hazard_score : 0-100 normalized incident density per localidad (min-max
                   rescaling over the valid localidades; NOT calibrated).
  - exposure_score : 0-100 = model probability * 100 (demo model on synthetic
                   BRFSS-like data).
  - score_integral : (hazard * exposure) / 100 -> 0-100 multiplicative.

IMPORTANT: the way the two components are combined is a PLACEHOLDER. The real
calibration of the combination belongs to the methodological design phase
(Hito 2). This is stated honestly in the UI.
"""

from __future__ import annotations

import data_nuse


NOTA_CALIBRACION = (
    "La combinacion de peligro y exposicion en un solo indice es un "
    "PLACEHOLDER de calibracion: la ponderacion/transformacion definitiva "
    "se definira en la fase metodologica (Hito 2). Los valores actuales "
    "sirven para demostrar el flujo integral del sistema, no son una "
    "medida de riesgo validada."
)

NOTA_DATOS = (
    "Datos de incidentes historicos de la linea 123 (NUSE) agregados por "
    "localidad, y modelo de exposicion DEMO entrenado sobre datos sinteticos "
    "tipo-BRFSS. Ninguno de los dos componentes esta aun validado para uso "
    "decisorio."
)


def hazard_score(localidad: str) -> dict:
    """Territorial hazard 0-100 for a localidad name (Module A)."""
    resumen = data_nuse.cargar_resumen()
    fila = resumen[resumen["LOCALIDAD"] == localidad]
    if fila.empty:
        raise KeyError(f"Localidad desconocida: {localidad}")
    fila = fila.iloc[0]
    return {
        "localidad": localidad,
        "peligro_0_100": float(fila["peligro_0_100"]),
        "total_incidentes": int(fila["total_incidentes"]),
        "proporcion": float(fila["proporcion"]),
    }


def exposure_score(probabilidad: float) -> float:
    """Personal exposure 0-100 from a model probability (Module B)."""
    p = min(1.0, max(0.0, float(probabilidad)))
    return round(p * 100, 2)


def score_integral(peligro_0_100: float, exposicion_0_100: float) -> dict:
    """Multiplicative integral score, normalized to 0-100."""
    peligro = min(100.0, max(0.0, float(peligro_0_100)))
    exposicion = min(100.0, max(0.0, float(exposicion_0_100)))
    integral = round((peligro * exposicion) / 100.0, 1)

    if integral < 33:
        banda, color = "Riesgo bajo", "#2E7D32"
    elif integral <= 66:
        banda, color = "Riesgo medio", "#C55A11"
    else:
        banda, color = "Riesgo alto", "#C00000"

    return {
        "integral_0_100": integral,
        "banda": banda,
        "color": color,
        "peligro_0_100": peligro,
        "exposicion_0_100": exposicion,
        "interpretacion": (
            f"Su localidad ({fila_localidad(peligro)}) aporta un peligro territorial "
            f"de {peligro:.0f}/100, y su perfil personal una exposicion de "
            f"{exposicion:.0f}/100. El score integral resultante es "
            f"{integral:.1f}/100 ({banda}). "
        ),
    }


def fila_localidad(peligro: float) -> str:
    """Helper returning a Spanish phrase by hazard band (used in text)."""
    if peligro >= 66:
        return "con alta densidad historica de incidentes"
    if peligro >= 33:
        return "con densidad historica de incidentes media"
    return "con densidad historica de incidentes baja"


def contexto_datos() -> str:
    """Provenance-driven caption for the UI."""
    prov = data_nuse.cargar_proveniencia()
    period = prov.get("periodo", {})
    return (
        f"Datos historicos NUSE (linea 123) {period.get('anio_min', '?')}-"
        f"{period.get('anio_max', '?')}; la barra de peligro usa densidad "
        "normalizada por localidad (min-max 0-100), placeholder de calibracion."
    )
