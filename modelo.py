"""Supervised model: predict monthly incident counts per UPZ.

Turns the descriptive (historical-frequency) map into a PREDICTIVE one. For each
UPZ and target month t we predict the incident count using ONLY prior months
(lag1, lag12, lag13, moving averages, month and year) -> no leakage.

Models compared (temporal split: train <= 2023-12, test 2024-01..2025-12):
    - baseline_lag1  : previous month
    - baseline_blend : mean(previous month, same month last year)
    - gbm            : GradientBoostingRegressor

At monthly granularity the model beats the naive baselines (see metrics).

Artifacts: data/modelo.joblib and data/modelo_meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import data_nuse

DATA_DIR = Path(__file__).resolve().parent / "data"
ARTEFACTO = DATA_DIR / "modelo.joblib"
META = DATA_DIR / "modelo_meta.json"

FEATURES = ["lag1", "lag12", "lag13", "ma3", "ma12", "MES", "ANIO"]
TARGET = "incidentes"
T_TRAIN_FIN = 2023 * 12 + 12      # through 2023-12
T_TEST_INI = 2024 * 12 + 1        # 2024-01
T_TEST_FIN = 2025 * 12 + 12       # 2025-12


def _t(anio: int, mes: int) -> int:
    return int(anio) * 12 + int(mes)


def serie_mensual() -> pd.DataFrame:
    hechos = data_nuse.cargar_hechos()
    return hechos.groupby(["COD_UPZ", "ANIO", "MES"], as_index=False)["total_incidentes"].sum()


def construir_panel() -> pd.DataFrame:
    """One row per (UPZ, target month) with leakage-free lag features."""
    serie = serie_mensual()
    serie["t"] = serie["ANIO"] * 12 + serie["MES"]
    t_min, t_max = int(serie["t"].min()), int(serie["t"].max())
    todos_t = list(range(t_min, t_max + 1))

    filas = []
    for upz, g in serie.groupby("COD_UPZ"):
        vals = dict(zip(g["t"].astype(int), g["total_incidentes"].astype(float)))
        for t in todos_t:
            if t - 12 < t_min or t - 1 < t_min:
                continue  # need at least a year of history
            filas.append({
                "COD_UPZ": upz,
                "ANIO": (t - 1) // 12,
                "MES": (t - 1) % 12 + 1,
                "t": t,
                "lag1": vals.get(t - 1, 0.0),
                "lag12": vals.get(t - 12, 0.0),
                "lag13": vals.get(t - 13, 0.0),
                "ma3": np.mean([vals.get(t - k, 0.0) for k in (1, 2, 3)]),
                "ma12": np.mean([vals.get(t - 12 - k, 0.0) for k in (0, 1, 2)]),
                TARGET: vals.get(t, 0.0),
            })
    return pd.DataFrame(filas)


def _metricas(y, y_pred) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    share = float(np.mean(np.abs(y / y.sum() - y_pred / y_pred.sum()))) * 100 if y.sum() > 0 else float("nan")
    return {
        "mae": round(float(mean_absolute_error(y, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 2),
        "share_mae_pct": round(share, 4),
    }


def entrenar(guardar: bool = True) -> dict:
    panel = construir_panel()
    train = panel[panel["t"] <= T_TRAIN_FIN]
    test = panel[(panel["t"] >= T_TEST_INI) & (panel["t"] <= T_TEST_FIN)]
    Xtr, ytr = train[FEATURES].to_numpy(), train[TARGET].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test[TARGET].to_numpy()

    gbm = GradientBoostingRegressor(random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05)
    gbm.fit(Xtr, ytr)

    preds = {
        "baseline_lag1": test["lag1"].to_numpy(),
        "baseline_blend": (test["lag1"].to_numpy() + test["lag12"].to_numpy()) / 2,
        "gbm": gbm.predict(Xte),
    }
    metricas = {n: _metricas(yte, p) for n, p in preds.items()}

    mejor = min(metricas, key=lambda k: metricas[k]["mae"])
    base_mae = metricas["baseline_lag1"]["mae"]
    mejora = round((base_mae - metricas[mejor]["mae"]) / base_mae * 100, 2) if base_mae else 0.0
    modelo_final = gbm if mejor == "gbm" else None
    importancias = ({f: round(float(v), 4) for f, v in zip(FEATURES, gbm.feature_importances_)}
                    if mejor == "gbm" else None)

    print(f"[modelo] muestras train/test: {len(train):,} / {len(test):,}")
    for n, m in metricas.items():
        print(f"[modelo] {n:16s} MAE={m['mae']:>8,.1f}  RMSE={m['rmse']:>8,.1f}  shareMAE={m['share_mae_pct']:.4f}%"
              + ("  <= mejor" if n == mejor else ""))
    print(f"[modelo] mejor: {mejor} | mejora vs baseline(lag1): {mejora}%")

    artefacto = {
        "modelo": modelo_final, "nombre": mejor, "features": FEATURES, "target": TARGET,
        "metricas": metricas, "mejora_vs_baseline_pct": mejora, "importancias": importancias,
    }
    if guardar:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefacto, ARTEFACTO)
        META.write_text(json.dumps({
            "modelo": mejor, "features": FEATURES, "metricas": metricas,
            "mejora_vs_baseline_pct": mejora, "importancias": importancias,
            "train": "hasta 2023-12", "test": "2024-01 .. 2025-12",
            "nota": "Split temporal mensual (sin fuga de futuro).",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[modelo] guardado en {ARTEFACTO.name}")
    return artefacto


_CARGADO = None


def cargar_modelo() -> dict:
    global _CARGADO
    if _CARGADO is None:
        _CARGADO = joblib.load(ARTEFACTO) if ARTEFACTO.exists() else entrenar(guardar=True)
    return _CARGADO


def metricas_guardadas() -> dict | None:
    return json.loads(META.read_text(encoding="utf-8")) if META.exists() else None


def predecir_mes(anio: int, mes: int) -> pd.DataFrame:
    """Predicted incident count per UPZ for the given month (history < month)."""
    artefacto = cargar_modelo()
    panel = construir_panel()
    fila = panel[panel["t"] == _t(anio, mes)].copy()
    if fila.empty:
        raise ValueError(f"Sin historia suficiente para predecir {anio}-{mes:02d}.")
    modelo = artefacto["modelo"]
    pred = fila["lag1"].to_numpy(dtype=float) if modelo is None else modelo.predict(fila[FEATURES].to_numpy())
    fila["predicho"] = np.clip(pred, 0, None)
    return fila[["COD_UPZ", "predicho"]]


def primer_mes_predecible() -> tuple[int, int]:
    """Earliest (year, month) the model can predict (needs >= 1 year of history)."""
    panel = construir_panel()
    t = int(panel["t"].min())
    return ((t - 1) // 12, (t - 1) % 12 + 1)


def ultimo_mes_predecible() -> tuple[int, int]:
    panel = construir_panel()
    t = int(panel["t"].max())
    return ((t - 1) // 12, (t - 1) % 12 + 1)


if __name__ == "__main__":
    entrenar()
