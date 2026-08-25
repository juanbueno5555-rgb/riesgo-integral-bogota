# RIESGO INTEGRAL · Bogotá — Prototipo MVP

Prototipo funcional para el proyecto académico PTIA (Escuela Colombiana de
Ingeniería, Ingeniería de Sistemas). El sistema calcula un **score de riesgo
integral** a partir de dos componentes, bajo el marco ISO 31000:

```
Score integral = Peligro (territorio) × Exposición (persona)
```

- **Módulo A — Peligro territorial**: densidad histórica de incidentes
  reportados a la línea de emergencias 123 (NUSE) de Bogotá. Se presenta como un
  mapa interactivo con calor de una muestra de puntos y con "zonas predichas"
  (k-means) marcadas claramente como **prototipo**.
- **Módulo B — Exposición personal**: perfil de estilo de vida (edad, IMC,
  hipertensión, colesterol, actividad física, tabaquismo, salud general) que
  devuelve una probabilidad estimada de enfermedad cardiometabólica (diabetes o
  enfermedad cardiovascular) mediante regresión logística.
- **Integración**: el usuario selecciona una localidad de Bogotá y el sistema
  combina el peligro territorial (0-100) con la exposición personal (0-100) en
  un score integral (0-100) con barras de desglose y texto de interpretación.

---

## Estado de los datos y del modelo

| Componente            | Estado actual                                             | Reemplazo futuro            |
| --------------------- | --------------------------------------------------------- | --------------------------- |
| Incidentes NUSE       | Agregados **reales** 2015-2026 (localidad × año × tipo)   | —                           |
| Puntos del mapa       | **Muestra sintética** con distribución basada en datos reales | Incidentes reales con coordenadas |
| Zonas predichas       | **Placeholder** k-means sobre puntos de muestra           | Modelo de predicción real   |
| Modelo de salud       | **Demo** entrenada sobre datos **sintéticos** tipo-BRFSS (AUC reportado) | BRFSS 2015 real            |
| Calibración del score | **Placeholder** de calibración (definición en Hito 2)     | Calibración metodológica    |

> Los valores del prototipo son demostrativos: **no son una medida de riesgo
> validada** y no deben usarse con fines de decisión o clínicos.

---

## Estructura del proyecto

```
prototipo/
├── app.py               # Aplicación Streamlit (interfaz)
├── data_nuse.py         # Carga ligera de los datos agregados NUSE
├── preparar_datos.py    # Script offline: genera los agregados desde el CSV crudo
├── modelo_riesgo.py     # Dataset sintético + regresión logística + predicción
├── integracion.py       # hazard_score / exposure_score / score_integral
├── smoke_check.py       # Verificación automatizada (sin Streamlit)
├── requirements.txt     # Dependencias
├── assets/
│   ├── logo.svg         # Logo vectorial (marca demo)
│   └── ico.svg          # Marca para favicon
└── data/                # Generado por preparar_datos.py (no incluye el CSV crudo)
```

## ¿Cómo ejecutar?

```bash
pip install -r requirements.txt          # 1. instalar dependencias
python preparar_datos.py                  # 2. generar agregados desde el CSV crudo
python smoke_check.py                     # 3. (opcional) verificación automatizada
streamlit run app.py                      # 4. lanzar la aplicación
```

> `preparar_datos.py` lee el CSV crudo de NUSE (113 MB) indicado por ruta
> (argumento opcional) o la ruta por defecto documentada. Solo se ejecuta una
> vez: sus salidas ligeras quedan en `data/`. La aplicación funciona sin el CSV
> crudo, usando únicamente los agregados.

## Notas metodológicas

- **Filtros**: los códigos `99`/`-` (SIN LOCALIZACIÓN) se excluyen de los
  agregados. La localidad Sumapaz (`20`) se conserva aunque aporta pocos
  incidentes. Todo queda documentado en `data/proveniencia.json`.
- **Normalización del peligro**: reescalado min-max 0-100 de los incidentes por
  localidad (densidad relativa). La combinación `(Peligro × Exposición) / 100`
  es un **placeholder de calibración** que se definirá formalmente en la fase
  metodológica (Hito 2).
- **Explicabilidad (Módulo B)**: contribución = coeficiente × (valor − media),
  etiquetada como demostración (demo XAI), no como XAI validado.

## Paleta de marca (demo)

Azul marino `#1F3864` · Azul corporativo `#2E75B6` · Acento cálido `#C55A11` ·
Alerta `#C00000`. El nombre de marca "RIESGO INTEGRAL" es un nombre provisional
que será reemplazado por el nombre final elegido por el equipo.

## Fases posteriores

- Incidentes reales georreferenciados (reemplazan los puntos de muestra).
- Modelo BRFSS 2015 real (reemplaza el modelo sintético).
- Modelo de predicción de zonas (reemplaza los centros k-means).
- Diseño metodológico y calibración formal del score integral (Hito 2).
