"""Sentinel · Mapa de probabilidad de incidentes — Bogotá (Streamlit MVP).

Proyecto académico PTIA (Escuela Colombiana de Ingeniería, Ingeniería de Sistemas).

A partir de los incidentes reales reportados a la Línea 123 (NUSE, 2015-2026),
agregados por UPZ, el sistema muestra DOS mapas de PROBABILIDAD por UPZ:

  - Real     : frecuencia relativa empírica del periodo seleccionado.
  - Predicho : estimación del modelo supervisado (GBM mensual) para el último
               mes del periodo seleccionado.

La probabilidad es relativa al histórico/predicción; no es una probabilidad
calibrada ni causal.
"""

from __future__ import annotations

import copy
from pathlib import Path

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap

import data_nuse
import mapa
import modelo

NAVY = "#1F3864"
BLUE = "#2E75B6"
ACENTO = "#C55A11"
ALERTA = "#C00000"
FONDO = "#000000"

RAIZ = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _superficie_historica(desde: tuple[int, int], hasta: tuple[int, int], tipo: str | None):
    return mapa.superficie_historica(desde, hasta, tipo)


@st.cache_data(show_spinner=False)
def _superficie_predicha(anio: int, mes: int):
    return mapa.superficie_predicha(anio, mes)


@st.cache_data(show_spinner=False)
def _geojson_base() -> dict:
    return data_nuse.cargar_geojson()


@st.cache_data(show_spinner=False)
def _localidad_base() -> dict:
    return data_nuse.cargar_localidad_geo()


@st.cache_data(show_spinner=False)
def _siniestros() -> pd.DataFrame:
    return data_nuse.cargar_siniestros()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def estilo_global() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {FONDO}; }}
        h1, h2, h3 {{ color: #FFFFFF; }}
        p, li, label {{ color: #FFFFFF; }}
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def inyectar_logo() -> None:
    ruta = RAIZ / "assets" / "logo.svg"
    if ruta.exists():
        st.markdown(f'<div style="text-align:center;padding:0.2rem 0;">{ruta.read_text(encoding="utf-8")}</div>',
                    unsafe_allow_html=True)


def sidebar() -> None:
    with st.sidebar:
        inyectar_logo()
        st.markdown("---")
        st.subheader("Avisos")
        st.markdown(
            '<div class="nota"><b>Datos reales:</b> incidentes reportados a la Línea 123 '
            "(NUSE, Bogotá) 2015-2026, agregados por UPZ. Fuente: Datos Abiertos Bogotá (SDSCJ).</div>",
            unsafe_allow_html=True)
        st.markdown(
            '<div class="nota"><b>Probabilidad:</b> el mapa <b>Real</b> usa la frecuencia relativa '
            "empírica del periodo; el mapa <b>Predicho</b> usa la estimación del modelo. No es calibrada ni causal.</div>",
            unsafe_allow_html=True)
        metrics = modelo.metricas_guardadas()
        if metrics:
            st.markdown("---")
            st.subheader("Modelo")
            mm = metrics["metricas"]
            st.caption(
                f"GBM mensual · MAE test **{mm['gbm']['mae']:,.0f}** vs baseline "
                f"{mm['baseline_lag1']['mae']:,.0f} (-{metrics['mejora_vs_baseline_pct']}%)"
            )
            if metrics.get("importancias"):
                st.caption("Importancia de features:")
                for k, v in sorted(metrics["importancias"].items(), key=lambda x: -x[1])[:4]:
                    st.caption(f"· {k}: {v}")
        st.markdown("---")
        st.caption("Proyecto académico PTIA · Escuela Colombiana de Ingeniería")
        st.caption("Derly Valeria Pachón Pinzón · Juan David Rangel Jiménez")


def header(superficie, periodo_txt: str, modo: str) -> None:
    prov = data_nuse.cargar_proveniencia()
    col_titulo, col_kpi = st.columns([2, 1])
    with col_titulo:
        st.markdown(
            '<div class="hero"><span class="badge">MVP</span>'
            f'<span class="badge">{modo}</span>'
            '<span class="badge">NUSE Línea 123</span></div>',
            unsafe_allow_html=True)
        st.title("Sentinel · Mapa de probabilidad de incidentes")
        st.markdown("Superficie de probabilidad por **UPZ** a partir de datos reales de la Línea 123.")
    with col_kpi:
        k1, k2, k3 = st.columns(3)
        k1.metric("UPZ", len(superficie))
        k2.metric("Incidentes", f"{int(superficie['total_incidentes'].sum()):,.0f}")
        k3.metric("Periodo", periodo_txt)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def panel_filtros():
    etiquetas = [data_nuse.etiqueta(a, m) for a, m in data_nuse.periodos()]
    c1, c2 = st.columns([3, 1])
    with c1:
        rango = st.select_slider("Línea de tiempo (meses)", options=etiquetas,
                                 value=(etiquetas[0], etiquetas[-1]))
    with c2:
        tipo_sel = st.selectbox("Tipo de incidente", ["Todos"] + data_nuse.listar_tipos(),
                                index=0,
                                help="Aplica al mapa Real. El mapa Predicho usa totales (sin filtro de tipo).")
    tipo = None if tipo_sel == "Todos" else tipo_sel
    return rango, tipo


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _bounds(geo: dict):
    lats, lons = [], []
    for f in geo.get("features", []):
        g = f["geometry"]
        poligonos = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in poligonos:
            for ring in poly:
                for x, y in ring:
                    lons.append(float(x)); lats.append(float(y))
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def _centroide(geom) -> tuple[float, float]:
    anillos = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = anillos[0][0]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (sum(ys) / len(ys), sum(xs) / len(xs))  # (lat, lon)


def construir_mapa(geo: dict, vmax: float, localidad_geo: dict | None = None,
                   siniestros: pd.DataFrame | None = None) -> folium.Map:
    colormap = cm.linear.YlOrRd_09.scale(0, vmax)
    colormap.caption = f"Probabilidad (%) · escala fija 0–{vmax:.1f}"
    mapa_f = folium.Map(location=[4.65, -74.10], zoom_start=11, tiles=None, control_scale=True)

    # --- Basemaps (seleccionables) ---
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
        name="Oscuro", show=True).add_to(mapa_f)
    folium.TileLayer("OpenStreetMap", name="Calles", show=False).add_to(mapa_f)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri, Maxar, Earthstar Geographics", name="Satélite", show=False).add_to(mapa_f)

    # --- Capa principal: probabilidad por UPZ ---
    folium.GeoJson(
        geo, name="Probabilidad por UPZ",
        style_function=lambda f: {"fillColor": colormap(float(f["properties"].get("prob_pct", 0.0))),
                                  "color": "#FFFFFF", "weight": 0.5, "fillOpacity": 0.82},
        highlight_function=lambda _: {"weight": 2.0, "color": "#FFFFFF", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["nombre_upz", "localidad", "total_incidentes", "prob_pct"],
            aliases=["UPZ:", "Localidad:", "Incidentes:", "Probabilidad (%):"],
            localize=True, sticky=True),
    ).add_to(mapa_f)

    # --- Capa: límites de localidad ---
    loc = (localidad_geo or {}).get("features", [])
    if loc:
        folium.GeoJson(
            localidad_geo, name="Límites de localidad", show=True,
            style_function=lambda _: {"fillOpacity": 0.0, "color": "#00E5FF", "weight": 1.6},
            tooltip=folium.GeoJsonTooltip(fields=["localidad"], aliases=["Localidad:"], sticky=True),
        ).add_to(mapa_f)

    # --- Capa: nombres de UPZ (apagada por defecto) ---
    nombres = folium.FeatureGroup(name="Nombres de UPZ", show=False)
    for f in geo.get("features", []):
        try:
            lat, lon = _centroide(f["geometry"])
        except Exception:  # noqa: BLE001
            continue
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=(
                f'<div style="font-size:8px;color:#FFF;text-shadow:0 0 2px #000;">'
                f'{f["properties"].get("nombre_upz", "")}</div>'), icon_size=(0, 0)),
        ).add_to(nombres)
    nombres.add_to(mapa_f)

    # --- Capa: siniestros viales georreferenciados (puntos reales, apagada) ---
    if siniestros is not None and not siniestros.empty:
        calor = folium.FeatureGroup(name="Siniestros viales (puntos)", show=False)
        HeatMap(
            siniestros[["lat", "lon"]].to_numpy().tolist(),
            radius=7, blur=9, min_opacity=0.25,
            gradient={0.2: "#2E75B6", 0.5: "#FFEB3B", 0.8: "#C55A11", 1.0: "#C00000"},
        ).add_to(calor)
        calor.add_to(mapa_f)

    mapa_f.fit_bounds(_bounds(geo))
    folium.LayerControl(collapsed=False).add_to(mapa_f)
    colormap.add_to(mapa_f)
    return mapa_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Sentinel · Mapa de probabilidad — Bogotá",
                       page_icon=":material/location_on:", layout="wide",
                       initial_sidebar_state="expanded")
    estilo_global()
    sidebar()
    try:
        data_nuse._asegurar_datos()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron cargar los datos: {exc}")
        st.stop()

    rango, tipo = panel_filtros()
    desde = _parse(rango[0]); hasta = _parse(rango[1])

    # Mapa REAL: frecuencia relativa empírica del periodo/tipo seleccionados.
    superficie_real = _superficie_historica(desde, hasta, tipo)
    periodo_real = rango[0] if rango[0] == rango[1] else f"{rango[0]}→{rango[1]}"
    etiqueta_real = f"Real · {periodo_real}" + (f" · {tipo}" if tipo else "")

    # Mapa PREDICHO: estimación del modelo para el mes objetivo del periodo.
    p0 = modelo.primer_mes_predecible()
    objetivo = hasta if hasta >= p0 else p0
    if objetivo != hasta:
        st.info(
            f"El modelo necesita al menos 1 año de historia previa. Se muestra la "
            f"predicción para **{data_nuse.etiqueta(*objetivo)}** (primer mes predecible)."
        )
    superficie_pred = _superficie_predicha(*objetivo)
    periodo_pred = data_nuse.etiqueta(*objetivo)

    header(superficie_real, periodo_real, "Real + Predicho")

    if int(superficie_real["total_incidentes"].sum()) == 0:
        st.warning("No hay datos para la selección. Probá otro rango o tipo.")
        st.stop()

    # Siniestros reales (capa opcional), acotados al periodo del mapa real.
    sin = _siniestros()
    y_ini, y_fin = max(desde[0], 2015), min(hasta[0], 2021)
    if not sin.empty and y_ini <= y_fin:
        sub = sin[sin["anio"].between(y_ini, y_fin)]
        sin_sel = sub.sample(min(len(sub), 30000), random_state=0) if len(sub) > 30000 else sub
    else:
        sin_sel = sin.iloc[0:0]

    vmax = mapa.escala_prob_max()
    loc = _localidad_base()
    geo_real = mapa.enriquecer_geojson(copy.deepcopy(_geojson_base()), superficie_real)
    geo_pred = mapa.enriquecer_geojson(copy.deepcopy(_geojson_base()), superficie_pred)
    mapa_real = construir_mapa(geo_real, vmax, loc, sin_sel)
    mapa_pred = construir_mapa(geo_pred, vmax, loc, sin_sel)

    st.subheader("Superficie de probabilidad — real vs predicho")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**Real (histórico)** · {periodo_real}" + (f" · {tipo}" if tipo else ""))
        components.html(mapa_real._repr_html_(), height=520, scrolling=False)
    with mc2:
        st.markdown(f"**Predicho (modelo)** · {periodo_pred}")
        components.html(mapa_pred._repr_html_(), height=520, scrolling=False)
    if not sin_sel.empty:
        st.caption(
            f"Capa «Siniestros viales (puntos)»: **{len(sin_sel):,} puntos reales** "
            f"georreferenciados ({y_ini}–{y_fin}), fuente SDM. Activala en el control de capas."
        )
    st.caption(
        "El color codifica la probabilidad (%) en una escala fija, comparable entre ambos mapas. "
        "Pasá el cursor sobre una UPZ para ver su detalle."
    )

    comp = mapa.comparacion_mes(*objetivo)
    mae = float(comp["error"].abs().mean())
    corr = float(comp["predicho"].corr(comp["real"]))
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("UPZ con mayor probabilidad (real)")
        st.bar_chart(mapa.top_upz(superficie_real, 10).set_index("nombre_upz")["prob_pct"], height=320)
    with r2:
        st.subheader(f"Predicho vs real — {periodo_pred}")
        top = comp.head(12).set_index("nombre_upz")[["predicho", "real"]]
        top.columns = ["Predicho", "Real"]
        st.bar_chart(top, height=320, stack=False)
    r3, r4 = st.columns(2)
    with r3:
        st.subheader("Dispersión por UPZ (112)")
        st.scatter_chart(comp, x="real", y="predicho", height=320)
    with r4:
        st.subheader("Incidentes por mes")
        pm = data_nuse.cargar_por_mes().copy()
        pm["fecha"] = pm["ANIO"].astype(str) + "-" + pm["MES"].astype(str).str.zfill(2)
        st.bar_chart(pm.set_index("fecha")["total_incidentes"], height=320)
    st.caption(
        f"MAE de {periodo_pred}: **{mae:,.1f}** incidentes por UPZ · "
        f"correlación predicho-real: **{corr:.3f}**. "
        "Puntos sobre la diagonal = predicción perfecta."
    )

    metrics = modelo.metricas_guardadas()
    if metrics:
        with st.expander("Detalle del modelo predictivo"):
            st.markdown(
                f"**Modelo:** GBM mensual · **train** {metrics['train']} · **test** {metrics['test']}\n\n"
                f"| Modelo | MAE | RMSE |\n|---|---|---|\n"
                f"| baseline (mes anterior) | {metrics['metricas']['baseline_lag1']['mae']:,.1f} | "
                f"{metrics['metricas']['baseline_lag1']['rmse']:,.1f} |\n"
                f"| blend (mes ant. + año ant.) | {metrics['metricas']['baseline_blend']['mae']:,.1f} | "
                f"{metrics['metricas']['baseline_blend']['rmse']:,.1f} |\n"
                f"| **GBM** | **{metrics['metricas']['gbm']['mae']:,.1f}** | "
                f"**{metrics['metricas']['gbm']['rmse']:,.1f}** |\n\n"
                f"El GBM mejora el MAE del baseline en **{metrics['mejora_vs_baseline_pct']}%**."
            )

    prov = data_nuse.cargar_proveniencia()
    st.markdown("---")
    st.caption(
        f"Fuente: {Path(prov['fuente_datos']).name} · {prov['periodo']['anio_min']}-"
        f"{prov['periodo']['anio_max']} · {prov['total_incidentes']:,} incidentes · {prov['upz_con_datos']} UPZ."
    )


def _parse(etiqueta: str) -> tuple[int, int]:
    a, m = etiqueta.split("-")
    return int(a), int(m)


if __name__ == "__main__":
    main()
