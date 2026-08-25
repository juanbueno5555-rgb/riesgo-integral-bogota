"""Demo risk model for personal exposure (lifestyle profile -> diabetes/CVD).

A logistic regression trained on a SYNTHETIC BRFSS-like dataset. This is a
DEMO model: the generated data is not real BRFSS data. A real model trained on
BRFSS 2015 will replace it in a later phase. Everything needed to swap the
model is contained in a single function (entrenar / cargar_modelo / predecir).

Artifacts are saved to prototipo/data/modelo_riesgo.joblib (pickle container
with model + metadata) and modelo_riesgo_meta.json (readable summary).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
ARTEFACTO_MODELO = DATA_DIR / "modelo_riesgo.joblib"
META_JSON = DATA_DIR / "modelo_riesgo_meta.json"

CARACTERISTICAS = [
    "edad", "bmi", "hipertension", "colesterol_alto",
    "actividad_fisica", "fumador", "salud_general",
]

# Neutral Spanish labels used by the UI for explainability output.
ETIQUETAS = {
    "edad": "Edad",
    "bmi": "Indice de masa corporal (IMC)",
    "hipertension": "Hipertension diagnosticada",
    "colesterol_alto": "Colesterol alto",
    "actividad_fisica": "Actividad fisica regular",
    "fumador": "Consumo de tabaco",
    "salud_general": "Salud general autopercibida",
}

N_FILAS = 5000
SEMILLA = 42
OBJETIVO = "diabetes_or_cvd"


# ----------------------------------------------------------------------------
# Synthetic dataset generation (DEMO)
# ----------------------------------------------------------------------------

def generar_datos_sinteticos(n: int = N_FILAS, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    n = int(n)

    edad = rng.integers(18, 81, size=n).astype(float)
    bmi = np.clip(rng.normal(27.0, 5.0, size=n), 15.0, 50.0)
    hipertension = rng.binomial(1, 0.30, size=n).astype(float)
    colesterol_alto = rng.binomial(1, 0.35, size=n).astype(float)
    actividad_fisica = rng.binomial(1, 0.55, size=n).astype(float)
    fumador = rng.binomial(1, 0.18, size=n).astype(float)
    salud_general = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.30, 0.35, 0.18, 0.07]).astype(float)

    X = np.column_stack([edad, bmi, hipertension, colesterol_alto,
                         actividad_fisica, fumador, salud_general])

    # Logistic relationship with realistic directions (higher = worse for all
    # except physical activity, which lowers risk).
    coeficientes = np.array([0.035, 0.06, 0.9, 0.7, -0.7, 0.6, 0.3])
    logit = -6.2 + X @ coeficientes + rng.normal(0, 0.4, size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    etiqueta = rng.binomial(1, prob).astype(int)

    df = pd.DataFrame(
        {
            "edad": edad, "bmi": bmi, "hipertension": hipertension,
            "colesterol_alto": colesterol_alto, "actividad_fisica": actividad_fisica,
            "fumador": fumador, "salud_general": salud_general,
            OBJETIVO: etiqueta,
        }
    )
    return df


# ----------------------------------------------------------------------------
# Training / persistence
# ----------------------------------------------------------------------------

def entrenar(guardar: bool = True) -> dict:
    """Train the logistic regression, print metrics, and persist the artifact.

    Returns a metrics dict (auc, n, prevalence, coefficients).
    """
    print("[modelo_riesgo] generando dataset sintetico tipo-BRFSS...")
    df = generar_datos_sinteticos()
    X = df[CARACTERISTICAS].to_numpy()
    y = df[OBJETIVO].to_numpy()

    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    modelo = LogisticRegression(max_iter=2000, random_state=SEMILLA)
    modelo.fit(X_ent, y_ent)

    prob_prueba = modelo.predict_proba(X_prueba)[:, 1]
    auc = float(roc_auc_score(y_prueba, prob_prueba))

    prev = float(y.mean())
    print(f"[modelo_riesgo] filas: {len(df)} | prevalencia objetivo: {prev:.1%}")
    print(f"[modelo_riesgo] AUC en conjunto de prueba (holdout): {auc:.4f}")
    print("[modelo_riesgo] coeficientes:")
    for nombre, coef in zip(CARACTERISTICAS, modelo.coef_[0]):
        print(f"    {nombre:>16s} = {coef:+.4f}")

    metricas = {
        "auc": round(auc, 4),
        "n_filas": int(len(df)),
        "n_prueba": int(len(X_prueba)),
        "prevalencia": round(prev, 4),
        "coeficientes": {nombre: round(float(coef), 4)
                         for nombre, coef in zip(CARACTERISTICAS, modelo.coef_[0])},
        "datos": "SINTETICOS tipo-BRFSS (demo) - no son datos reales BRFSS",
        "modelo": "Regresion logistica",
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    }

    if guardar:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        artefacto = {
            "modelo": modelo,
            "features": CARACTERISTICAS,
            "medias": {nombre: float(df[nombre].mean()) for nombre in CARACTERISTICAS},
            "desvios": {nombre: float(df[nombre].std()) for nombre in CARACTERISTICAS},
            "metricas": metricas,
        }
        joblib.dump(artefacto, ARTEFACTO_MODELO)
        with open(META_JSON, "w", encoding="utf-8") as fh:
            json.dump(metricas, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[modelo_riesgo] modelo guardado en {ARTEFACTO_MODELO.name}")

    return metricas


_ARTEFACTO_CARGADO = None


def cargar_modelo() -> dict:
    """Load the model artifact, training it first if necessary."""
    global _ARTEFACTO_CARGADO
    if _ARTEFACTO_CARGADO is not None:
        return _ARTEFACTO_CARGADO
    if ARTEFACTO_MODELO.exists():
        _ARTEFACTO_CARGADO = joblib.load(ARTEFACTO_MODELO)
    else:
        print("[modelo_riesgo] artefacto ausente; entrenando modelo demo...")
        entrenar()
        _ARTEFACTO_CARGADO = joblib.load(ARTEFACTO_MODELO)
    return _ARTEFACTO_CARGADO


# ----------------------------------------------------------------------------
# Prediction + demo explainability (coefficient-based)
# ----------------------------------------------------------------------------

def predecir(perfil: dict) -> dict:
    """Prediction + demo XAI for a single profile dict.

    perfil keys: edad, bmi, hipertension (0/1), colesterol_alto (0/1),
                 actividad_fisica (0/1), fumador (0/1), salud_general (1..5).

    Returns dict with probabilidad, exposicion_0_100 and top contribuciones.
    """
    artefacto = cargar_modelo()
    modelo = artefacto["modelo"]
    features = artefacto["features"]
    medias = artefacto["medias"]

    fila = np.array([float(perfil[nombre]) for nombre in features]).reshape(1, -1)
    prob = float(modelo.predict_proba(fila)[0, 1])

    contribuciones = []
    coefs = modelo.coef_[0]
    for i, nombre in enumerate(features):
        contribuc = float(coefs[i] * (fila[0, i] - medias[nombre]))
        contribuciones.append(
            {
                "feature": nombre,
                "etiqueta": ETIQUETAS.get(nombre, nombre),
                "valor": round(float(fila[0, i]), 2),
                "contribucion": round(contribuc, 4),
            }
        )
    contribuciones.sort(key=lambda c: abs(c["contribucion"]), reverse=True)

    return {
        "probabilidad": prob,
        "exposicion_0_100": min(100.0, max(0.0, prob * 100)),
        "contribuciones": contribuciones,
    }


def metricas_guardadas() -> dict | None:
    """Read persisted metrics (AUC, etc.) from the meta JSON, if present."""
    if not META_JSON.exists():
        return None
    with open(META_JSON, encoding="utf-8") as fh:
        return json.load(fh)
