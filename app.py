"""RIESGO INTEGRAL · Bogotá — Streamlit MVP (PTIA, Escuela Colombiana de Ingeniería).

Score integral = Peligro (territorio, NUSE linea 123) × Exposicion (perfil de
estilo de vida, modelo demo) bajo el marco ISO 31000.

This is an academic prototype: data are historical aggregates plus synthetic
sample points, the health model is a DEMO trained on synthetic data, and the
calibration of the combined score is a placeholder until the methodological
design phase (Hito 2).
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

import data_nuse
import integracion
import modelo_riesgo

# Brand palette -----------------------------------------------------------------
NAVY = "#1F3864"
BLUE = "#2E75B6"
ACENTO = "#C55A11"
ALERTA = "#C00000"
FONDO = "#000000"
OK = "#2E7D32"

RAIZ = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RIESGO INTEGRAL · Bogotá",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------------------

def estilo_global() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --navy: {NAVY}; --blue: {BLUE}; --acento: {ACENTO};
        }}
        .stApp {{ background-color: {FONDO}; }}
        h1, h2, h3 {{ color: #FFFFFF; }}
        p, li {{ color: #FFFFFF; }}
        .hero {{ padding: 0.4rem 0 0.2rem 0; }}
        .badge {{
            display: inline-block; background: {BLUE}; color: #FFFFFF;
            padding: 0.15rem 0.7rem; border-radius: 14px;
            font-size: 0.8rem; margin-right: 0.35rem;
        }}
        .nota {{
            background: #161616; border-left: 4px solid {ACENTO};
            color: #FFFFFF; padding: 0.65rem 0.85rem; border-radius: 6px;
            font-size: 0.9rem; margin: 0.6rem 0;
        }}
        .alerta {{
            background: #2A1010; border-left: 4px solid {ALERTA};
            color: #FFFFFF; padding: 0.65rem 0.85rem; border-radius: 6px;
            font-size: 0.9rem; margin: 0.6rem 0;
        }}
        .barra {{ margin: 0.5rem 0; }}
        .barra .fila {{
            display: flex; justify-content: space-between;
            font-size: 0.88rem; font-weight: 600; color: #FFFFFF;
        }}
        .barra .pista {{
            background: #2A2A2A; border-radius: 8px; height: 14px; overflow: hidden;
        }}
        .barra .relleno {{ height: 100%; border-radius: 8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def barra(etiqueta: str, valor: float, color: str) -> str:
    valor = min(100.0, max(0.0, float(valor)))
    return (
        f'<div class="barra">'
        f'  <div class="fila"><span>{etiqueta}</span><span>{valor:.1f} / 100</span></div>'
        f'  <div class="pista"><div class="relleno" style="width:{valor:.1f}%;'
        f'background:{color};"></div></div>'
        f"</div>"
    )


def inyectar_logo() -> None:
    ruta = RAIZ / "assets" / "logo.svg"
    if not ruta.exists():
        return
    svg = ruta.read_text(encoding="utf-8")
    st.markdown(
        f'<div style="text-align:center;padding:0.2rem 0;">{svg}</div>',
        unsafe_allow_html=True,
    )


def cargar_datos_seguro() -> bool:
    try:
        data_nuse._asegurar_datos()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron cargar los datos agregados: {exc}")
        return False


# -------------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------------

def sidebar() -> None:
    with st.sidebar:
        inyectar_logo()
        st.markdown("---")
        st.markdown(
            f'<div style="color:#FFFFFF;font-weight:700;font-size:0.95rem;">Marca demo</div>'
            '<div style="color:#BBBBBB;font-size:0.82rem;">Paleta: azul marino '
            f'<span style="color:{NAVY};">●</span> azul corporativo '
            f'<span style="color:{BLUE};">●</span> acento cálido '
            f'<span style="color:{ACENTO};">●</span> alerta '
            f'<span style="color:{ALERTA};">●</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.subheader("Avisos importantes")
        st.markdown(
            f'<div class="nota"><b>Datos de incidentes (Módulo A):</b> agregados históricos '
            "de la línea 123 (NUSE) 2015-2026. Los puntos del mapa son una muestra "
            "sintética con distribución basada en los datos reales; los incidentes "
            "reales georreferenciados llegarán en una fase posterior.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="nota"><b>Modelo de salud (Módulo B):</b> demostración entrenada '
            "sobre datos <b>sintéticos</b> tipo-BRFSS. El modelo real sobre BRFSS 2015 "
            "llegará en una fase posterior.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="alerta"><b>Calibración del score integral:</b> {integracion.NOTA_CALIBRACION}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("Proyecto académico PTIA · Escuela Colombiana de Ingeniería · Ingeniería de Sistemas")
        st.caption("Integrantes: estudiantes (2) · Docente")


# -------------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------------

def header() -> pd.DataFrame | None:
    col_titulo, col_kpi = st.columns([2, 1])
    with col_titulo:
        st.markdown(
            '<div class="hero"><span class="badge">PROTOTIPO MVP</span>'
            '<span class="badge">Riesgo = Peligro × Exposición</span>'
            '<span class="badge">ISO 31000</span></div>',
            unsafe_allow_html=True,
        )
        st.title("RIESGO INTEGRAL · Bogotá")
        st.markdown(
            "Mapa interactivo de peligro territorial (Módulo A) combinado con el "
            "perfil de estilo de vida (Módulo B) para obtener un score integral."
        )
    resumen = data_nuse.cargar_resumen() if cargar_datos_seguro() else None
    if resumen is not None and not resumen.empty:
        prov = data_nuse.cargar_proveniencia()
        periodo = prov["periodo"]
        metricas_m = modelo_riesgo.metricas_guardadas() or {}
        auc = metricas_m.get("auc", "—")
        with col_kpi:
            k1, k2, k3 = st.columns(3)
            k1.metric("Localidades", len(resumen))
            k2.metric("Incidentes", f"{resumen['total_incidentes'].sum():,.0f}")
            k3.metric("Periodo", f"{periodo['anio_min']}-{periodo['anio_max']}")
            st.caption(f"AUC del modelo demo (holdout): {auc}")
    return resumen


# -------------------------------------------------------------------------------
# Módulo A : Mapa de riesgo territorial
# -------------------------------------------------------------------------------

def cargar_clusters(puntos: pd.DataFrame, n: int = 6) -> list[list[float]]:
    """k-means centers over the GIVEN points (follows the active filter)."""
    n = min(n, len(puntos))
    if n < 1:
        return []
    kmeans = KMeans(n_clusters=n, random_state=7, n_init=10)
    kmeans.fit(puntos[["lat", "lon"]].to_numpy())
    return kmeans.cluster_centers_.tolist()


def modulo_mapa(resumen: pd.DataFrame) -> None:
    st.header("Mapa de riesgo territorial")
    st.caption(
        "Peligro del territorio a partir de la densidad histórica de incidentes "
        "reportados a la línea 123 (NUSE)."
    )

    try:
        puntos = data_nuse.cargar_puntos()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron cargar los puntos de muestra: {exc}")
        return

    # Filter by localidad CODE (stable), display by name: avoids any
    # accent/case mismatch between derived files.
    nombre_a_codigo = dict(zip(resumen["LOCALIDAD"], resumen["COD_LOCALIDAD"]))
    opciones = ["Todas las localidades"] + list(nombre_a_codigo)
    filtro = st.selectbox("Filtrar localidad", opciones, index=0)

    df_mapa = puntos
    if filtro != "Todas las localidades":
        cod = nombre_a_codigo[filtro]
        df_mapa = puntos[puntos["cod_localidad"].astype(str) == str(cod)]

    if df_mapa.empty:
        st.warning(
            "No hay puntos de muestra para la selección actual (la localidad "
            "puede tener muy pocos incidentes). Pruebe con otra localidad."
        )
        return

    mapa = folium.Map(location=[4.66, -74.10], zoom_start=11, tiles="OpenStreetMap")
    HeatMap(
        df_mapa[["lat", "lon"]].values.tolist(),
        radius=13,
        blur=16,
        min_opacity=0.35,
        gradient={0.0: "#2E75B6", 0.5: "#FFEB3B", 0.75: "#C55A11", 1.0: "#C00000"},
    ).add_to(mapa)

    # Predicted zones follow the current filter (all points, or the localidad).
    centros_cluster = cargar_clusters(df_mapa)
    for lat, lon in centros_cluster:
        folium.CircleMarker(
            location=[lat, lon],
            radius=13,
            color="#FFFFFF",
            weight=3,
            fill=True,
            fill_color="#C55A11",
            fill_opacity=0.95,
            popup="Zona predicha (PROTOTIPO) - centro de grupo k-means",
            tooltip="Zona predicha (PROTOTIPO)",
        ).add_to(mapa)

    # Zoom to the selected localidad so the filter is visible immediately.
    if filtro != "Todas las localidades" and len(df_mapa) > 1:
        mapa.fit_bounds(
            [
                [df_mapa["lat"].min(), df_mapa["lon"].min()],
                [df_mapa["lat"].max(), df_mapa["lon"].max()],
            ]
        )

    # Plain HTML embed: pan/zoom stay 100% client-side (no Streamlit re-runs).
    components.html(mapa._repr_html_(), height=540)

    st.markdown(
        f'<div class="nota"><b>Lectura del mapa:</b> el calor muestra la densidad '
        "de la <b>muestra de puntos sintética</b> cuya distribución por localidad se "
        "construyó a partir de los <b>incidentes reales</b> de la línea 123. "
        f"<b>Zonas predichas:</b> {len(centros_cluster)} centros k-means calculados "
        "sobre los puntos visibles (PROTOTIPO); el modelo de predicción real de "
        "zonas llegará en una fase posterior.</div>",
        unsafe_allow_html=True,
    )

    col_top, col_tipos = st.columns(2)
    with col_top:
        st.subheader("Localidades con más incidentes")
        top = resumen.head(10).set_index("LOCALIDAD")["total_incidentes"]
        st.bar_chart(top, height=300)
    with col_tipos:
        st.subheader("Tipos de incidentes más frecuentes")
        tipos = data_nuse.cargar_top_tipos(10)
        tabla = tipos.copy()
        tabla.columns = ["Tipo de incidente", "Total"]
        st.dataframe(tabla, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------------
# Módulo B : Perfil de estilo de vida
# -------------------------------------------------------------------------------

def valores_a_perfil(edad, bmi, hipertension, colesterol, actividad, fumador, salud) -> dict:
    return {
        "edad": float(edad),
        "bmi": float(bmi),
        "hipertension": 1 if hipertension == "Sí" else 0,
        "colesterol_alto": 1 if colesterol == "Sí" else 0,
        "actividad_fisica": 1 if actividad == "Sí" else 0,
        "fumador": 1 if fumador == "Sí" else 0,
        "salud_general": float(salud),
    }


def renderizar_resultado_perfil() -> None:
    if "perfil_prob" not in st.session_state:
        return
    prob = st.session_state["perfil_prob"]
    exp = st.session_state["perfil_exp"]
    factores = st.session_state.get("perfil_factores", [])

    st.markdown("### Resultado de su perfil")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Probabilidad estimada (demo)**")
        st.markdown(
            f"<p style='font-size:2.4rem;font-weight:700;color:#FFFFFF;line-height:1;'>"
            f"{prob:.1%}</p><p>de presentar enfermedad cardiometabólica "
            "(diabetes o enfermedad cardiovascular).</p>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Su exposición personal (0-100)**")
        st.markdown(barra("Exposición (Módulo B)", exp, BLUE), unsafe_allow_html=True)

    st.markdown("**Factores que más contribuyen a su resultado (explicabilidad demo)**")
    for f in factores[:5]:
        direccion = "aumenta" if f["contribucion"] > 0 else "disminuye"
        color = ALERTA if f["contribucion"] > 0 else OK
        st.markdown(
            f"· {f['etiqueta']}: <span style='color:{color};font-weight:600;'>"
            f"{direccion}</span> su exposición (coef. {f['contribucion']:+.3f})",
            unsafe_allow_html=True,
        )
    st.caption(
        "Explicabilidad demo basada en coeficientes de la regresión logística "
        "(contribución = coef × (valor − media)). No es XAI validado."
    )


def modulo_perfil() -> None:
    st.header("Mi perfil de estilo de vida")
    st.caption(
        "Exposición personal (Módulo B): probabilidad estimada de diabetes o "
        "enfermedad cardiovascular a partir del estilo de vida."
    )
    st.warning(
        "Este modelo es una DEMO entrenada sobre datos SINTÉTICOS tipo-BRFSS, "
        "no sobre datos reales. El modelo real (BRFSS 2015) llegará en una fase "
        "posterior; los resultados no deben usarse con fines clínicos."
    )

    with st.form("perfil"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", min_value=18, max_value=90, value=45, step=1)
            bmi = st.number_input("IMC (índice de masa corporal)", min_value=15.0, max_value=50.0, value=26.0, step=0.5)
            salud = st.selectbox("Salud general autopercibida", [1, 2, 3, 4, 5], index=2, format_func=lambda x: {1: "1 · Excelente", 2: "2 · Muy buena", 3: "3 · Buena", 4: "4 · Regular", 5: "5 · Mala"}[x])
        with c2:
            hipertension = st.selectbox("Hipertensión diagnosticada", ["No", "Sí"])
            colesterol = st.selectbox("Colesterol alto", ["No", "Sí"])
            actividad = st.selectbox("Actividad física regular", ["No", "Sí"])
            fumador = st.selectbox("Consumo de tabaco", ["No", "Sí"])
        enviar = st.form_submit_button("Calcular mi exposición", type="primary")

    if enviar:
        perfil = valores_a_perfil(edad, bmi, hipertension, colesterol, actividad, fumador, salud)
        with st.spinner("Calculando exposición..."):
            resultado = modelo_riesgo.predecir(perfil)
        st.session_state["perfil_prob"] = resultado["probabilidad"]
        st.session_state["perfil_exp"] = resultado["exposicion_0_100"]
        st.session_state["perfil_factores"] = resultado["contribuciones"]
        st.session_state["perfil_perfil"] = perfil

    renderizar_resultado_perfil()

    if st.button("Usar perfil de ejemplo", help="Carga un perfil de referencia para probar el score integral."):
        ejemplo = valores_a_perfil(55, 29.0, "No", "Sí", "No", "No", 3)
        resultado = modelo_riesgo.predecir(ejemplo)
        st.session_state["perfil_prob"] = resultado["probabilidad"]
        st.session_state["perfil_exp"] = resultado["exposicion_0_100"]
        st.session_state["perfil_factores"] = resultado["contribuciones"]
        st.session_state["perfil_perfil"] = ejemplo
        st.rerun()


# -------------------------------------------------------------------------------
# Integración : Score integral
# -------------------------------------------------------------------------------

def modulo_score(resumen: pd.DataFrame) -> None:
    st.header("Score integral = Peligro × Exposición")
    st.caption(
        "Combinación de la densidad histórica de incidentes (territorio) con el "
        "perfil personal de salud, bajo el marco ISO 31000 (Riesgo = Peligro × Exposición)."
    )

    localidades = data_nuse.listar_localidades()
    localidad = st.selectbox(
        "Seleccione su localidad de Bogotá",
        localidades,
        index=0,
        help="Localidad donde habita o transita con mayor frecuencia.",
    )

    try:
        peligro = integracion.hazard_score(localidad)
    except KeyError as exc:
        st.error(f"Localidad no encontrada: {exc}")
        return

    tiene_perfil = "perfil_exp" in st.session_state
    if tiene_perfil:
        exposicion = integracion.exposure_score(st.session_state["perfil_prob"])
    else:
        st.info(
            "Aún no calculó su perfil personal. Puede calcularlo en la pestaña "
            "«Mi perfil de estilo de vida» o usar el perfil de ejemplo."
        )
        if st.button("Usar perfil de ejemplo para el score integral"):
            ejemplo = valores_a_perfil(55, 29.0, "No", "Sí", "No", "No", 3)
            resultado = modelo_riesgo.predecir(ejemplo)
            st.session_state["perfil_prob"] = resultado["probabilidad"]
            st.session_state["perfil_exp"] = resultado["exposicion_0_100"]
            st.session_state["perfil_factores"] = resultado["contribuciones"]
            st.session_state["perfil_perfil"] = ejemplo
            st.rerun()
        return

    integral = integracion.score_integral(peligro["peligro_0_100"], exposicion)

    st.markdown("### Desglose del score")
    st.markdown(
        barra("Peligro territorial (Módulo A)", peligro["peligro_0_100"], NAVY),
        unsafe_allow_html=True,
    )
    st.markdown(
        barra("Exposición personal (Módulo B)", exposicion, BLUE),
        unsafe_allow_html=True,
    )
    st.markdown(
        barra("Score integral", integral["integral_0_100"], integral["color"]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Resultado", f"{integral['integral_0_100']:.1f} / 100", integral["banda"])
        st.caption(
            f"{peligro['total_incidentes']:,} incidentes históricos en {localidad} "
            f"({peligro['proporcion']:.1%} del total de Bogotá)."
        )
    with c2:
        st.markdown(
            f'<div class="alerta"><b>Interpretación ({integral["banda"]}):</b> '
            f"{integral['interpretacion']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Fórmula del prototipo: Score = (Peligro × Exposición) / 100. "
            f"{integracion.NOTA_CALIBRACION}"
        )

    st.markdown("---")
    st.caption(f"Contexto de datos: {integracion.contexto_datos()}")


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------

def main() -> None:
    estilo_global()
    sidebar()
    if not cargar_datos_seguro():
        st.stop()
    resumen = header()

    tab_mapa, tab_perfil, tab_score = st.tabs(
        ["Mapa de riesgo territorial", "Mi perfil de estilo de vida", "Score integral"]
    )
    with tab_mapa:
        modulo_mapa(resumen)
    with tab_perfil:
        modulo_perfil()
    with tab_score:
        modulo_score(resumen)

    st.markdown("---")
    st.caption(
        "Prototipo MVP · Proyecto académico PTIA · Escuela Colombiana de Ingeniería. "
        "Datos de muestra, modelo sintético y calibración placeholder; "
        "reemplazos reales en fases posteriores del proyecto."
    )


if __name__ == "__main__":
    main()
