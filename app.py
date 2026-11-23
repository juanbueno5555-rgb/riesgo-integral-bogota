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

import base64
import copy
import json
from pathlib import Path

import branca.colormap as cm
import folium
from folium.plugins import HeatMap
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import data_nuse
import mapa
import modelo

NAVY = "#1F3864"
BLUE = "#2E75B6"
ACENTO = "#C55A11"
ALERTA = "#C00000"

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
def _superficie_pronosticada(anio: int, mes: int):
    return mapa.superficie_pronosticada(anio, mes)


@st.cache_data(show_spinner=False)
def _pronostico_meta() -> dict:
    """Curva de error por horizonte y pronóstico mes a mes (artefacto del script)."""
    ruta = RAIZ / "data" / "pronostico_meta.json"
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def _tabla_pronostico(ini: int, fin: int) -> pd.DataFrame:
    """Pronóstico de los meses [ini, fin] (la página actual) con su error esperado."""
    meta = _pronostico_meta()
    err = {c["horizonte"]: c for c in meta.get("error_por_horizonte", [])}
    filas = []
    for f in meta.get("pronostico", [])[ini - 1:fin]:
        e = err.get(f["horizonte"], {})
        filas.append({
            "Mes": _nombre_desde(f["mes"]),
            "Horizonte": f"h={f['horizonte']}",
            "Total proyectado": f"{int(f['total_predicho']):,}",
            "Error esperado": (f"±{f['error_total_pct']:.1f} %"
                               if f.get("error_total_pct") is not None else "s/d"),
            "Error por UPZ": (f"±{e['wape_pct']:.1f} %"
                              if e.get("wape_pct") is not None else "s/d"),
            "Casos": e.get("muestras", "-"),
        })
    return pd.DataFrame(filas)


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
        .hero {{ padding: 0.4rem 0 0.2rem 0; }}
        .badge {{
            display: inline-block; background: {BLUE}; color: #FFFFFF;
            padding: 0.15rem 0.7rem; border-radius: 14px;
            font-size: 0.8rem; margin-right: 0.35rem;
        }}
        .nota {{
            background: var(--secondary-background-color);
            border-left: 4px solid {ACENTO};
            color: var(--text-color);
            padding: 0.65rem 0.85rem; border-radius: 6px;
            font-size: 0.9rem; margin: 0.6rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inyectar_logo() -> None:
    ruta = RAIZ / "assets" / "logo_sentinel.jpeg"
    if ruta.exists():
        b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
        st.markdown(
            '<div style="text-align:center;padding:0.2rem 0;">'
            f'<img src="data:image/jpeg;base64,{b64}" alt="Sentinel" '
            'style="width:100%;max-width:260px;border-radius:8px;"/></div>',
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

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
POR_PAGINA = 12


def _nombre_mes(anio: int, mes: int) -> str:
    return f"{MESES[int(mes) - 1]} {int(anio)}"


def _nombre_desde(etiqueta_txt: str) -> str:
    a, m = etiqueta_txt.split("-")
    return _nombre_mes(int(a), int(m))


def _selector_futuro(t_ult: int) -> int:
    """Paginación de meses futuros; devuelve el horizonte elegido (1..HORIZONTE_MAX)."""
    total = modelo.HORIZONTE_MAX
    paginas = (total + POR_PAGINA - 1) // POR_PAGINA
    pagina = int(st.session_state.get("pagina_futuro", 0))
    ini = pagina * POR_PAGINA + 1
    fin = min(ini + POR_PAGINA - 1, total)

    nav_prev, nav_info, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("◀ 12 meses", disabled=pagina == 0):
            st.session_state["pagina_futuro"] = max(0, pagina - 1)
            st.rerun()
    with nav_info:
        a, b = _nombre_mes(*_parse(modelo._etiqueta(t_ult + ini))), \
            _nombre_mes(*_parse(modelo._etiqueta(t_ult + fin)))
        st.markdown(
            f"<div style='text-align:center;padding-top:8px'>"
            f"<b>Página {pagina + 1} de {paginas}</b> · {a} — {b}</div>",
            unsafe_allow_html=True)
    with nav_next:
        if st.button("12 meses ▶", disabled=fin >= total):
            st.session_state["pagina_futuro"] = pagina + 1
            st.rerun()

    etiquetas = {_nombre_mes(*_parse(modelo._etiqueta(t_ult + k))): k
                 for k in range(ini, fin + 1)}
    sel = st.segmented_control("Mes a pronosticar", list(etiquetas),
                               selection_mode="single")
    return etiquetas.get(sel, ini)


def panel_filtros():
    publicados = [data_nuse.etiqueta(a, m) for a, m in data_nuse.periodos()]
    t_ult = modelo.ultimo_t_publicado()
    primer_pred = modelo.primer_mes_predecible()
    predecibles = [e for e in publicados if _parse(e) >= primer_pred]

    c1, c2, c3 = st.columns([3, 1, 1])
    with c3:
        modo = st.radio("Vista", ["Histórico", "Predicho", "Predicho a futuro"],
                        horizontal=False)

    horizonte = 0
    with c1:
        if modo == "Histórico":
            rango = st.select_slider("Rango acumulado (histórico)", options=publicados,
                                     value=(publicados[0], publicados[-1]), key="rango_hist")
        elif modo == "Predicho":
            mes = st.select_slider("Mes a comparar (ya publicado)", options=predecibles,
                                   value=predecibles[-1], key="mes_pred")
            rango = (mes, mes)
        else:
            rango = (publicados[0], publicados[-1])
    with c2:
        deshabilitado = modo != "Histórico"
        tipo_sel = st.selectbox("Tipo de incidente", ["Todos"] + data_nuse.listar_tipos(),
                                index=0, disabled=deshabilitado,
                                help="Las vistas de predicción usan totales (sin filtro de tipo).")
    tipo = None if (tipo_sel == "Todos" or deshabilitado) else tipo_sel
    if modo == "Predicho a futuro":
        horizonte = _selector_futuro(t_ult)
    return rango, modo, tipo, horizonte


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
                   siniestros: pd.DataFrame | None = None,
                   puntos_pred: pd.DataFrame | None = None) -> folium.Map:
    colormap = cm.linear.YlOrRd_09.scale(0, vmax)
    colormap.caption = f"Probabilidad (%) · escala fija 0–{vmax:.1f}"
    mapa_f = folium.Map(location=[4.635, -74.115], zoom_start=11, tiles=None, control_scale=True)

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

    # --- Capa de límites de localidad: eliminada a pedido (generaba las líneas cyan) ---

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

    # --- Capa: siniestros viales (mapa de calor, apagada por defecto) ---
    # leaflet.heat dibuja el radio en píxeles fijos: de lejos los puntos se funden
    # en una mancha y de cerca se ven como motas. El script del final recalcula el
    # radio en cada zoom para que la lectura se mantenga a cualquier escala.
    if siniestros is not None and not siniestros.empty:
        capa = folium.FeatureGroup(name="Siniestros viales (puntos)", show=False)
        HeatMap(
            siniestros[["lat", "lon"]].to_numpy().tolist(),
            radius=8, blur=5, min_opacity=0.05,
            gradient={0.2: "#2E75B6", 0.5: "#FFEB3B", 0.8: "#C55A11", 1.0: "#C00000"},
        ).add_to(capa)
        capa.add_to(mapa_f)

    # --- Capa: puntos de predicción (uno por UPZ, tamaño por magnitud) ---
    if puntos_pred is not None and not puntos_pred.empty:
        puntos = folium.FeatureGroup(name="Puntos de predicción", show=True)
        por_cod = {str(f["properties"].get("cod_upz", "")).strip().upper(): f["geometry"]
                   for f in geo.get("features", [])}
        vmax_p = float(puntos_pred["total_incidentes"].max()) or 1.0
        for _, r in puntos_pred.iterrows():
            geom = por_cod.get(str(r["COD_UPZ"]).strip().upper())
            if geom is None:
                continue
            lat, lon = _centroide(geom)
            val = float(r["total_incidentes"])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3 + 11 * (val / vmax_p) ** 0.5,
                color="#FFD166", weight=0.8,
                fill=True, fillColor="#FF8C42", fillOpacity=0.8,
                tooltip=f"{r['nombre_upz']}: {val:,.0f} proyectados",
            ).add_to(puntos)
        puntos.add_to(mapa_f)

    # Sin fit_bounds: se ejecuta antes de que el contenedor tenga su tamaño final
    # y a veces encuadra lejos o deformado. El centro y el zoom son fijos.
    mapa_f.get_root().html.add_child(folium.Element(_js_zoom_calor(mapa_f.get_name())))
    folium.LayerControl(collapsed=False).add_to(mapa_f)
    return mapa_f


def _js_zoom_calor(nombre_mapa: str) -> str:
    """Radio del heatmap en función del zoom (el nativo es fijo y se ve mal)."""
    return f"""<script>
(function () {{
  var NAME = "{nombre_mapa}";
  var intentos = 0;
  function buscar(map) {{
    var encontrado = null;
    map.eachLayer(function (l) {{ if (l && l._heat) {{ encontrado = l; }} }});
    return encontrado;
  }}
  function ajustar(map) {{
    var capa = buscar(map);
    if (!capa) {{ return false; }}
    var z = map.getZoom();
    var radio = Math.min(34, Math.max(4, Math.round(7 * Math.pow(1.7, z - 11))));
    capa.setOptions({{ radius: radio, blur: Math.round(radio * 0.6) }});
    return true;
  }}
  function iniciar() {{
    var map = window[NAME];
    if (!map) {{
      if (intentos++ < 80) {{ setTimeout(iniciar, 100); }}
      return;
    }}
    map.on("zoomend", function () {{ ajustar(map); }});
    map.on("layeradd", function () {{ setTimeout(function () {{ ajustar(map); }}, 40); }});
    ajustar(map);
    if (!buscar(map) && intentos++ < 80) {{ setTimeout(iniciar, 100); }}
  }}
  if (document.readyState !== "loading") {{ setTimeout(iniciar, 0); }}
  else {{ document.addEventListener("DOMContentLoaded", function () {{ setTimeout(iniciar, 0); }}); }}
}})();
</script>"""


def leyenda(vmax: float) -> None:
    """Barra de color limpia, fuera del mapa (legible y sin tapar la superficie)."""
    st.markdown(
        '<div style="margin:2px 0 10px 0;">'
        '<div style="height:16px;border-radius:8px;'
        'background:linear-gradient(90deg,#ffffcc,#ffeda0,#fed976,#feb24c,'
        '#fd8d3c,#fc4e2a,#e31a1c,#b10026);"></div>'
        '<div style="display:flex;justify-content:space-between;color:var(--text-color);'
        'font-size:0.72rem;margin-top:3px;">'
        f'<span>0.0 %</span><span>{vmax / 2:.1f} %</span><span>{vmax:.1f} %</span></div>'
        f'<div style="color:var(--text-color);opacity:0.7;font-size:0.72rem;">Probabilidad (%) · escala fija 0–{vmax:.1f}</div>'
        '</div>',
        unsafe_allow_html=True)


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

    rango, modo, tipo, horizonte = panel_filtros()
    desde = _parse(rango[0]); hasta = _parse(rango[1])

    # Mapa REAL: frecuencia relativa empírica del periodo/tipo seleccionados.
    superficie_real = _superficie_historica(desde, hasta, tipo)
    periodo_real = rango[0] if rango[0] == rango[1] else f"{rango[0]}→{rango[1]}"

    header(superficie_real, periodo_real, modo)

    if int(superficie_real["total_incidentes"].sum()) == 0:
        st.warning("No hay datos para la selección. Probá otro rango o tipo.")
        st.stop()

    # Siniestros reales (capa opcional), acotados al periodo.
    sin = _siniestros()
    y_ini, y_fin = max(desde[0], 2015), min(hasta[0], 2021)
    if not sin.empty and y_ini <= y_fin:
        sub = sin[sin["anio"].between(y_ini, y_fin)]
        sin_sel = sub.sample(min(len(sub), 15000), random_state=0) if len(sub) > 30000 else sub
    else:
        sin_sel = sin.iloc[0:0]

    vmax = mapa.escala_prob_max()
    loc = _localidad_base()
    geo_real = mapa.enriquecer_geojson(copy.deepcopy(_geojson_base()), superficie_real)
    mapa_real = construir_mapa(geo_real, vmax, loc, sin_sel)

    if modo == "Predicho a futuro":
        t_ult = modelo.ultimo_t_publicado()
        h = max(1, horizonte)
        objetivo = _parse(modelo._etiqueta(t_ult + h))
        periodo_pred = _nombre_mes(*objetivo)
        superficie_pred = _superficie_pronosticada(*objetivo)
        geo_pred = mapa.enriquecer_geojson(copy.deepcopy(_geojson_base()), superficie_pred)
        mapa_pred = construir_mapa(geo_pred, vmax, loc, sin_sel, puntos_pred=superficie_pred)
        total_pred = int(superficie_pred["total_incidentes"].sum())

        curva = {c["horizonte"]: c for c in _pronostico_meta().get("error_por_horizonte", [])}
        err = curva.get(h)
        st.warning(
            f"**Pronóstico · {periodo_pred}** — {h} mes{'es' if h > 1 else ''} desde "
            f"{modelo._etiqueta(t_ult)}. "
            + (f"Error esperado del **total mensual: ±{err['error_total_pct']:.1f} %** · "
               f"por UPZ ±{err['wape_pct']:.1f} % · medido sobre {err['muestras']} casos."
               if err else "Error no medido para este horizonte.")
        )

        st.markdown(f"**Pronóstico (modelo)** · {periodo_pred}")
        components.html(mapa_pred._repr_html_(), height=560, scrolling=False)
        leyenda(vmax)

        pagina = int(st.session_state.get("pagina_futuro", 0))
        ini_pag = pagina * POR_PAGINA + 1
        fin_pag = min(ini_pag + POR_PAGINA - 1, modelo.HORIZONTE_MAX)
        st.subheader(f"Pronóstico mes a mes · {_nombre_desde(modelo._etiqueta(t_ult + ini_pag))}"
                     f" — {_nombre_desde(modelo._etiqueta(t_ult + fin_pag))}")
        st.dataframe(_tabla_pronostico(ini_pag, fin_pag), hide_index=True, width="stretch")

        col_top, col_total = st.columns([2, 1])
        with col_top:
            st.subheader("UPZ con mayor incidencia proyectada")
            st.bar_chart(
                mapa.top_upz(superficie_pred, 10).set_index("nombre_upz")["prob_pct"],
                height=320)
        with col_total:
            st.metric("Total proyectado", f"{total_pred:,}",
                      help="Suma de las 112 UPZ para el mes seleccionado.")
            st.caption("Los puntos del mapa son la predicción por UPZ. El error crece con el "
                       "horizonte porque cada paso usa la predicción anterior como entrada.")

    elif modo == "Predicho":
        p0 = modelo.primer_mes_predecible()
        objetivo = hasta if hasta >= p0 else p0
        if objetivo != hasta:
            st.info(
                f"El modelo necesita al menos 1 año de historia previa. Se muestra la "
                f"predicción para **{data_nuse.etiqueta(*objetivo)}** (primer mes predecible)."
            )
        superficie_pred = _superficie_predicha(*objetivo)
        periodo_pred = data_nuse.etiqueta(*objetivo)
        geo_pred = mapa.enriquecer_geojson(copy.deepcopy(_geojson_base()), superficie_pred)
        mapa_pred = construir_mapa(geo_pred, vmax, loc, sin_sel, puntos_pred=superficie_pred)

        st.subheader("Predicción — real vs predicho")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**Real (histórico)** · {periodo_real}" + (f" · {tipo}" if tipo else ""))
            components.html(mapa_real._repr_html_(), height=520, scrolling=False)
        with mc2:
            st.markdown(f"**Predicho (modelo)** · {periodo_pred}")
            components.html(mapa_pred._repr_html_(), height=520, scrolling=False)
        leyenda(vmax)
        if not sin_sel.empty:
            st.caption(
                f"Capa «Siniestros viales (puntos)»: {len(sin_sel):,} puntos reales "
                f"({y_ini}-{y_fin}), fuente SDM. Activala en el control de capas."
            )

        comp = mapa.comparacion_mes(*objetivo)
        mae = float(comp["error"].abs().mean())
        corr = float(comp["predicho"].corr(comp["real"]))
        r1, r2 = st.columns(2)
        with r1:
            st.subheader(f"Predicho vs real — {periodo_pred}")
            top = comp.head(12).set_index("nombre_upz")[["predicho", "real"]]
            top.columns = ["Predicho", "Real"]
            st.bar_chart(top, height=320, stack=False)
        with r2:
            st.subheader("Dispersión por UPZ (112)")
            st.scatter_chart(comp, x="real", y="predicho", height=320)
        st.caption(
            f"MAE de {periodo_pred}: **{mae:,.1f}** incidentes por UPZ · "
            f"correlación predicho-real: **{corr:.3f}**. Puntos sobre la diagonal = predicción perfecta."
        )

    else:
        st.subheader("Superficie de probabilidad — histórico")
        components.html(mapa_real._repr_html_(), height=580, scrolling=False)
        leyenda(vmax)
        if not sin_sel.empty:
            st.caption(
                f"Capa «Siniestros viales (puntos)»: **{len(sin_sel):,} puntos reales** "
                f"georreferenciados ({y_ini}–{y_fin}), fuente SDM. Activala en el control de capas."
            )
        col_top, col_tend = st.columns(2)
        with col_top:
            st.subheader("UPZ con mayor probabilidad")
            st.bar_chart(mapa.top_upz(superficie_real, 10).set_index("nombre_upz")["prob_pct"], height=320)
        with col_tend:
            st.subheader("Incidentes por mes")
            pm = data_nuse.cargar_por_mes().copy()
            pm["fecha"] = pm["ANIO"].astype(str) + "-" + pm["MES"].astype(str).str.zfill(2)
            st.bar_chart(pm.set_index("fecha")["total_incidentes"], height=320)

    st.caption("El color codifica la probabilidad (%) en una escala fija. Pasá el cursor sobre una UPZ para ver su detalle.")

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
