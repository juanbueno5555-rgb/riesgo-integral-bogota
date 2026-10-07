# Sentinel · Mapa de probabilidad de incidentes — Bogotá

Prototipo funcional para el proyecto académico PTIA (Escuela Colombiana de
Ingeniería, Ingeniería de Sistemas). El sistema construye una **superficie de
probabilidad de incidentes por UPZ** a partir de los incidentes reales
reportados a la **Línea 123 (NUSE)** de Bogotá, en dos vistas:

- **Histórico** — frecuencia relativa empírica del periodo seleccionado.
- **Predicho** — estimación de un **modelo supervisado (GBM mensual)** para el
  último mes del periodo seleccionado.

> Alcance: **enfocado en el mapa** (según indicación del docente). El módulo de
> riesgo de salud individual quedó fuera de esta iteración.

---

## Qué hace

- Agrega **1.109.242 registros** NUSE (2015–2026) por **UPZ × año × mes × tipo**.
- Calcula la **probabilidad empírica** por UPZ (participación en el total).
- **Modelo predictivo** (`modelo.py`): GradientBoostingRegressor que estima los
  incidentes mensuales por UPZ usando solo historia previa (sin fuga de futuro).
- Pinta un **mapa coroplético** de las 112 UPZ con **línea de tiempo (rango de
  meses)**, filtro por tipo y toggle **Histórico / Predicho**.
- Muestra la **tendencia mensual** y el **ranking de UPZ**.

---

## El modelo (componente de IA)

- **Problema:** predicción de **conteos** (regresión) por UPZ-mes — supervisado.
- **Variables (features):** `lag1` (mes anterior), `lag12`/`lag13` (mismo mes del
  año anterior), medias móviles (`ma3`, `ma12`), `MES` y `ANIO`.
- **Validación temporal:** train hasta **2023-12**, test **2024-01 … 2025-12**
  (nunca split aleatorio, para no filtrar el futuro).
- **Comparación contra baselines** (mes anterior; blend mes/año anterior).

| Modelo | MAE | RMSE |
| --- | ---: | ---: |
| baseline (mes anterior) | 231.9 | 381.7 |
| blend (mes ant. + año ant.) | 191.3 | 297.4 |
| **GBM (elegido)** | **185.7** | **279.1** |

El GBM mejora el MAE del baseline en **≈19.9 %** → el modelo **sí aporta** a
granularidad mensual.

> Hallazgo honesto: con granularidad **anual** ningún modelo supera a la
> persistencia (el año anterior); por eso se adoptó la granularidad mensual.

---

## Estado de los datos

| Componente        | Estado                                                     |
| ----------------- | ---------------------------------------------------------- |
| Incidentes NUSE   | **Reales** (Datos Abiertos Bogotá, SDSCJ) 2015–2026         |
| Polígonos UPZ     | **Reales** (Datos Abiertos Bogotá) — 112 UPZ                |
| Probabilidad      | Histórico: frecuencia empírica · Predicho: modelo GBM        |

---

## Estructura del proyecto

```
prototipo/
├── app.py               # Aplicación Streamlit (mapa, timeline, toggle)
├── mapa.py              # Superficie de probabilidad (histórica y predicha)
├── modelo.py            # Modelo supervisado mensual (GBM) + evaluación
├── data_nuse.py         # Carga ligera de los datos agregados
├── preparar_datos.py    # Script offline: del CSV crudo a los agregados por UPZ
├── smoke_check.py       # Verificación automatizada (sin Streamlit)
├── requirements.txt     # Dependencias
├── assets/              # Logo (marca demo)
└── data/                # Generado por preparar_datos.py / modelo.py
    ├── upz_probabilidad.csv    # UPZ: total, prob_pct, indice_0_100
    ├── upz_anio_mes_tipo.csv.gz# tabla de hechos mensual (gzip)
    ├── nuse_por_anio.csv       # total por año
    ├── nuse_por_mes.csv        # total por mes
    ├── upz_geo.geojson         # polígonos UPZ (reproyectados a lon/lat)
    ├── modelo.joblib           # modelo entrenado
    ├── modelo_meta.json        # métricas del modelo
    └── proveniencia.json       # trazabilidad de la fuente y los filtros
```

## ¿Cómo ejecutar?

```bash
pip install -r requirements.txt          # 1. dependencias
python preparar_datos.py                  # 2. agregados desde el CSV crudo
python modelo.py                          # 3. entrenar/evaluar el modelo
python smoke_check.py                     # 4. (opcional) verificación
streamlit run app.py                      # 5. lanzar la aplicación
```

`preparar_datos.py` espera el CSV crudo de NUSE en `../datos/nuse_llamadas.csv` y
el GeoJSON de UPZ en `../datos/upz.geojson` (o se le pasan por argumento).

**Fuente de datos:**
- NUSE Línea 123 (SDSCJ, Datos Abiertos Bogotá) — ~115 MB, 1.109.242 filas.
- Polígonos UPZ (Datos Abiertos Bogotá) — 112 UPZ (EPSG:3857, reproyectados a WGS84).
- Licencia: Creative Commons Attribution Share-Alike 4.0.

## Notas metodológicas

- **Filtros:** se excluyen códigos sin localización (`99`, `-`, `UPZ999`) y las
  UPZ especiales/rurales que no están en el GeoJSON oficial.
- **Probabilidad:** `prob_pct = incidentes_UPZ / total × 100` sobre el periodo
  (o la predicción). El color usa un índice min-max 0–100.
- **Join espacial:** por `COD_UPZ` (formato `UPZnn`).

## Limitaciones (transparencia)

- Es un **sesgo de reporte**: son incidentes *reportados* a la Línea 123.
- La probabilidad es **relativa** al histórico/predicción; no es calibrada ni causal.
- El modelo usa solo la serie histórica por UPZ (no variables exógenas como
  población o comercio).
- Sin validez para decisiones de seguridad pública ni uso oficial.

## Fases posteriores

- Variables exógenas (población, comercio, iluminación) para mejorar el modelo.
- Manejo explícito de eventos atípicos (p. ej. confinamiento 2020).
- Probabilidad calibrada con intervalos de confianza.
