"""Smoke check for the Sentinel probability-map MVP (no Streamlit required).

Verifies:
  1. aggregated UPZ data present and sane (>=100 UPZ, shares sum ~100)
  2. monthly fact table present with the full year/month range
  3. UPZ GeoJSON present, joinable and with lon/lat inside Bogotá
  4. timeline periods available and ordered
  5. historical surface over a period/type is consistent (shares sum ~100)
  6. the supervised model trains and BEATS the naive baseline (monthly)
  7. predicted surface for a target month is valid
  8. provenance note is readable

Exits 0 only if every check passes.
"""

from __future__ import annotations

import data_nuse
import mapa
import modelo


def main() -> int:
    print("=== SMOKE CHECK: SENTINEL MAPA DE PROBABILIDAD ===")
    ok = fail = 0

    def check(nombre, fn):
        nonlocal ok, fail
        try:
            fn(); print(f"OK  {nombre}"); ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {nombre}: {exc}"); fail += 1

    def c1():
        upz = data_nuse.cargar_upz()
        assert len(upz) >= 100
        assert {"COD_UPZ", "nombre_upz", "localidad", "total_incidentes", "prob_pct"}.issubset(upz.columns)
        assert abs(upz["prob_pct"].sum() - 100.0) < 0.5
        print(f"    {len(upz)} UPZ, {int(upz['total_incidentes'].sum()):,} incidentes")

    def c2():
        h = data_nuse.cargar_hechos()
        assert len(h) > 50000, len(h)
        assert {"ANIO", "MES", "TIPO_DETALLE", "total_incidentes"}.issubset(h.columns)
        assert h["MES"].between(1, 12).all()
        print(f"    {len(h):,} filas mensuales, años {int(h['ANIO'].min())}-{int(h['ANIO'].max())}, {h['TIPO_DETALLE'].nunique()} tipos")

    def c3():
        geo = data_nuse.cargar_geojson()
        feats = geo["features"]
        assert len(feats) >= 100
        xs, ys = [], []
        for f in feats:
            co = f["geometry"]["coordinates"]
            polys = co if f["geometry"]["type"] == "MultiPolygon" else [co]
            for poly in polys:
                for ring in poly:
                    for x, y in ring:
                        xs.append(x); ys.append(y)
        assert -74.3 < min(xs) and max(xs) < -73.9, "lon fuera de Bogotá"
        assert 4.4 < min(ys) and max(ys) < 4.95, "lat fuera de Bogotá"
        print(f"    {len(feats)} polígonos con coords lon/lat en Bogotá")

    def c4():
        p = data_nuse.periodos()
        assert len(p) > 100
        assert p == sorted(p)
        print(f"    {len(p)} meses: {data_nuse.etiqueta(*p[0])} .. {data_nuse.etiqueta(*p[-1])}")

    def c5():
        p = data_nuse.periodos()
        full = mapa.superficie_historica(p[0], p[-1], None)
        assert len(full) == len(data_nuse.cargar_upz())
        assert abs(full["prob_pct"].sum() - 100.0) < 0.5
        corto = mapa.superficie_historica(p[-12], p[-1], None)
        assert 0 < int(corto["total_incidentes"].sum()) < int(full["total_incidentes"].sum())
        tipo = data_nuse.listar_tipos(1)[0]
        con_tipo = mapa.superficie_historica(p[0], p[-1], tipo)
        assert 0 < int(con_tipo["total_incidentes"].sum()) < int(full["total_incidentes"].sum())
        print(f"    total={int(full['total_incidentes'].sum()):,} | últimos 12m={int(corto['total_incidentes'].sum()):,} | '{tipo}'={int(con_tipo['total_incidentes'].sum()):,}")

    def c6():
        art = modelo.entrenar(guardar=True)
        m = art["metricas"]
        assert m["gbm"]["mae"] < m["baseline_lag1"]["mae"], (
            f"GBM ({m['gbm']['mae']}) no supera baseline ({m['baseline_lag1']['mae']})")
        assert art["mejora_vs_baseline_pct"] > 0
        print(f"    GBM MAE={m['gbm']['mae']:,.1f} < baseline {m['baseline_lag1']['mae']:,.1f} "
              f"(-{art['mejora_vs_baseline_pct']}%)")

    def c7():
        p0 = modelo.primer_mes_predecible()
        assert p0 == (2016, 1), f"primer mes predecible inesperado: {p0}"
        sup0 = mapa.superficie_predicha(*p0)
        assert len(sup0) == len(data_nuse.cargar_upz())
        assert abs(sup0["prob_pct"].sum() - 100.0) < 0.5
        p = data_nuse.periodos()
        a, m = p[-1]
        sup = mapa.superficie_predicha(a, m)
        assert len(sup) == len(data_nuse.cargar_upz())
        assert abs(sup["prob_pct"].sum() - 100.0) < 0.5
        assert (sup["total_incidentes"] >= 0).all() and sup["total_incidentes"].sum() > 0
        top = sup.iloc[0]
        print(f"    predicho {data_nuse.etiqueta(*p0)} y {data_nuse.etiqueta(a, m)}: top {top['nombre_upz']} ({top['prob_pct']:.2f}%)")

    def c8():
        prov = data_nuse.cargar_proveniencia()
        assert prov["periodo"]["anio_min"] and prov["periodo"]["anio_max"]
        print(f"    {prov['total_incidentes']:,} incidentes, {prov['upz_con_datos']} UPZ")

    def c9():
        sin = data_nuse.cargar_siniestros()
        assert len(sin) > 100000, f"pocos puntos de siniestros: {len(sin)}"
        assert sin["lat"].between(4.4, 4.95).all() and sin["lon"].between(-74.3, -73.9).all()
        loc = data_nuse.cargar_localidad_geo()
        assert len(loc.get("features", [])) == 20, "localidades != 20"
        print(f"    {len(sin):,} puntos de siniestros ({int(sin['anio'].min())}-{int(sin['anio'].max())}) "
              f"| {len(loc['features'])} localidades")

    check("datos UPZ", c1)
    check("tabla mensual (UPZ x mes x tipo)", c2)
    check("geojson UPZ (coords lon/lat)", c3)
    check("periodos (timeline)", c4)
    check("superficie histórica (periodo/tipo)", c5)
    check("modelo supera al baseline", c6)
    check("superficie predicha", c7)
    check("proveniencia", c8)
    check("siniestros viales + localidades", c9)

    print("=" * 48)
    print(f"SMOKE CHECK: {ok} OK, {fail} FAIL")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
