"""Dashboard Streamlit del predictor del Alavés (SPEC §9, F6).

Capa de presentación: toda la lógica está en `alaves_predictor.dashboard.data`
y en los módulos de modelos/simulación. Ejecutar con:

    uv run streamlit run app/dashboard.py

Páginas: próxima jornada, clasificación proyectada, el Alavés en detalle,
explicabilidad, rendimiento del modelo y registro de decisiones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alaves_predictor.config import load_settings
from alaves_predictor.dashboard import data as dd
from alaves_predictor.etl import db
from alaves_predictor.explain import importance
from alaves_predictor.features.build import build_features
from alaves_predictor.features.dictionary import describe
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANT_WITH_ODDS
from alaves_predictor.models.train import load_latest_model
from alaves_predictor.simulation.project import project_standings

st.set_page_config(page_title="Predictor Alavés — LaLiga", page_icon="⚽", layout="wide")


@st.cache_resource
def _settings():
    return load_settings(Path("config"))


@st.cache_resource
def _bundle_and_features():
    """Carga el modelo y construye el feature store una vez (cacheado)."""
    settings = _settings()
    conn = db.connect(settings.data.db_path)
    try:
        bundle = load_latest_model(conn)
        features = build_features(conn, settings, include_scheduled=True)
    finally:
        conn.close()
    return bundle, features


def _pct(col: pd.Series) -> pd.Series:
    return (col * 100).round(1)


def _season_selector(features: pd.DataFrame, settings) -> tuple[str, int | None]:
    """Elige temporada y, en modo demo (histórica), la jornada de corte."""
    seasons = sorted(features["season"].unique())
    current = settings.current_season
    default = seasons.index(current) if current in seasons else len(seasons) - 1
    season = st.sidebar.selectbox("Temporada", seasons, index=default)
    season_df = features[features["season"] == season]
    has_scheduled = season_df["result"].isna().any()
    if has_scheduled:
        return season, None  # temporada en curso: jugado vs programado real
    st.sidebar.caption(
        "Temporada histórica: modo demo. Elige desde qué jornada proyectar "
        "(las anteriores cuentan como reales)."
    )
    mds = dd.available_matchdays(features, season)
    from_md = st.sidebar.slider("Proyectar desde la jornada", min(mds) + 1, max(mds), max(mds) // 2)
    return season, from_md


def page_next_matchday(bundle, features, settings):
    st.header("Próxima jornada")
    if bundle is None:
        st.warning("No hay modelo entrenado. Ejecuta `alaves train`.")
        return
    season, _ = _season_selector(features, settings)
    mds = dd.available_matchdays(features, season)
    col_md, col_var = st.columns([2, 3])
    matchday = col_md.selectbox("Jornada", mds, index=0)
    variant = col_var.radio(
        "Variante del modelo",
        [VARIANT_NO_ODDS, VARIANT_WITH_ODDS],
        format_func=lambda v: (
            "Sin cuotas (interpretable)" if v == VARIANT_NO_ODDS else "Con cuotas"
        ),
        horizontal=True,
        help="La variante sin cuotas ignora el mercado; la de con cuotas es el techo de precisión.",
    )
    if variant not in bundle.variants:
        st.warning(f"El modelo entrenado no incluye la variante '{variant}'.")
        return

    def predict(rows):
        return bundle.predict_matches(rows, variant)

    preds = dd.matchday_predictions(predict, features, settings, season, matchday)
    if preds.empty:
        st.info("No hay partidos para esa jornada.")
        return

    # Selector de partido: por defecto el del equipo foco, si juega esa jornada
    focus = settings.focus_team
    labels = [f"{r.Local} vs {r.Visitante}" for r in preds.itertuples()]
    focus_mask = (preds["home_id"] == focus) | (preds["away_id"] == focus)
    default = int(focus_mask.to_numpy().argmax()) if focus_mask.any() else 0
    chosen = st.selectbox("Partido", labels, index=default)
    p = preds.iloc[labels.index(chosen)]

    st.subheader(f"{p['Local']} vs {p['Visitante']} — Jornada {matchday}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(victoria local)", f"{p['p_home'] * 100:.1f}%")
    c2.metric("P(empate)", f"{p['p_draw'] * 100:.1f}%")
    c3.metric("P(victoria visitante)", f"{p['p_away'] * 100:.1f}%")
    c4.metric("Marcador más probable", p["pred_score"])
    if p["Real"] != "—":
        st.caption(f"Resultado real: **{p['Real']}** (partido ya jugado)")
    fig = go.Figure(
        go.Bar(
            x=[f"1 · {p['Local']}", "X · Empate", f"2 · {p['Visitante']}"],
            y=[p["p_home"], p["p_draw"], p["p_away"]],
            marker_color=["#16a34a", "#94a3b8", "#dc2626"],
            text=[f"{v * 100:.1f}%" for v in (p["p_home"], p["p_draw"], p["p_away"])],
            textposition="outside",
            hovertemplate="%{x}: <b>%{y:.1%}</b><extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Resto de la jornada")
    cols = ["Local", "Visitante", "Predicho", "p_home", "p_draw", "p_away", "pred_score", "Real"]
    table = preds[cols].rename(
        columns={"p_home": "P(1)", "p_draw": "P(X)", "p_away": "P(2)", "pred_score": "Marcador"}
    )
    for c in ("P(1)", "P(X)", "P(2)"):
        table[c] = _pct(table[c])
    st.dataframe(table, width="stretch", hide_index=True)


# Zonas de la clasificación: etiqueta corta (para el gráfico), color fuerte
# (franjas y chips) y pastel (filas de tabla). Cuatro tonos bien separados —
# verde oscuro, lima, cian y rojo — para que Europa y Conference no se
# confundan. Los rangos de puestos vienen de config ([league.zones]).
_ZONE_STYLE = {
    "champions": ("Champions", "#15803d", "#bbf7d0"),
    "europa": ("Europa", "#84cc16", "#e3f7b8"),
    "conference": ("Conf.", "#0891b2", "#cffafe"),
    "descenso": ("Descenso", "#dc2626", "#fecaca"),
}

# Nombre completo (incluye `titulo`, que no pinta franja propia por ser un
# subconjunto de champions).
_ZONE_LABELS = {
    "titulo": "Título",
    "champions": "Champions",
    "europa": "Europa League",
    "conference": "Conference",
    "descenso": "Descenso",
}

# Escala de probabilidad en grises-violeta: neutra a propósito, para que el
# verde/rojo de las zonas destaque y no compita con el color de las celdas.
_PROB_SCALE = [
    [0.0, "#f8fafc"],
    [0.15, "#e2e8f0"],
    [0.4, "#a5b4fc"],
    [0.7, "#6366f1"],
    [1.0, "#312e81"],
]


def _style_projection_table(display, zone_by_row):
    """Colorea cada fila con el pastel de su zona y formatea los números.

    `zone_by_row` es la serie de zonas alineada con `display` (viene del PUESTO
    proyectado, no de la posición esperada: así hay exactamente 3 descendidos).
    """

    def color_row(row):
        zone = zone_by_row.get(row.name)
        pastel = _ZONE_STYLE[zone][2] if zone in _ZONE_STYLE else ""
        css = f"background-color: {pastel}; color: #111827" if pastel else ""
        return [css] * len(row)

    # Un único .format(): llamarlo dos veces resetea el formato de la primera
    # llamada (Styler reaplica el display por defecto a las columnas no citadas).
    fmt = {c: "{:.1%}" for c in display.columns if c.startswith("P(")}
    fmt.update({"Pts esperados": "{:.1f}", "Pos esperada": "{:.1f}º"})
    fmt = {c: f for c, f in fmt.items() if c in display.columns}
    return display.style.apply(color_row, axis=1).format(fmt)


def _zone_bands(zones, n_positions):
    """Zonas visibles (recortadas al tamaño de la liga) como (label, color, low, high)."""
    bands = []
    for key, (label, color, _pastel) in _ZONE_STYLE.items():
        rng = zones.get(key)
        if not rng:
            continue
        low, high = max(1, rng[0]), min(n_positions, rng[1])
        if low <= high:  # la zona cae dentro de la liga
            bands.append((label, color, low, high))
    return bands


def _projection_heatmap(heat, table, focus_name, zones, show_pct=True):
    """Heatmap de distribución de posiciones, con zonas y posición esperada.

    heat: DataFrame (filas = equipos ordenados por posición esperada, columnas =
    puestos 1..N, valores = probabilidad). Las zonas se pintan como FRANJAS
    VERTICALES que cruzan todo el mapa (verde Europa, rojo descenso), así se ve
    de un vistazo qué parte de la distribución de cada equipo cae en cada zona.
    """
    teams = list(heat.index)
    positions = list(heat.columns)
    z = heat.to_numpy()
    ylabels = [f"★ {t}" if t == focus_name else t for t in teams]
    # % solo en las celdas con probabilidad relevante (una tabla 20×20 con todo
    # el texto satura); el resto se lee por color y por el tooltip
    threshold = 0.12
    text = [[f"{v:.0%}" if v >= threshold else "" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=positions,
            y=ylabels,
            text=text if show_pct else None,
            texttemplate="%{text}" if show_pct else None,
            textfont={"size": 11, "color": "#f8fafc"},
            colorscale=_PROB_SCALE,
            zmin=0.0,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y}</b><br>Puesto %{x}º: %{z:.1%}<extra></extra>",
            colorbar={"title": "prob.", "tickformat": ".0%", "thickness": 12},
        )
    )
    expected = table.set_index("Equipo")["Pos esperada"].reindex(teams).to_numpy()
    fig.add_scatter(
        x=expected,
        y=ylabels,
        mode="markers",
        showlegend=False,
        marker={
            "symbol": "diamond",
            "size": 10,
            "color": "#f59e0b",
            "line": {"width": 1.5, "color": "#78350f"},
        },
        hovertemplate="%{y}: posición esperada %{x:.1f}º<extra></extra>",
    )

    # Franjas verticales de zona: cruzan todo el mapa y tiñen las columnas.
    # Las etiquetas se escalonan en altura porque zonas de un solo puesto
    # (Europa 6º, Conference 7º) quedan pegadas y sus textos se solaparían.
    for i, (label, color, low, high) in enumerate(_zone_bands(zones, len(positions))):
        fig.add_vrect(
            x0=low - 0.5,
            x1=high + 0.5,
            fillcolor=color,
            opacity=0.25,
            layer="above",
            line={"color": color, "width": 2},
            annotation_text=f"<b>{label}</b>",
            annotation_position="top",
            annotation={"font": {"size": 11, "color": color}, "yshift": 8 + 18 * (i % 2)},
        )

    # Rango exacto de las filas: sin esto el área del gráfico sobra por abajo y
    # las franjas tiñen ese hueco, que parece una casilla extra.
    fig.update_yaxes(range=[len(teams) - 0.5, -0.5])
    fig.update_xaxes(title="posición final", dtick=1, range=[0.5, len(positions) + 0.5])
    fig.update_layout(
        height=max(400, 30 * len(teams)),
        margin={"l": 10, "r": 10, "t": 72, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _team_distribution_chart(dist, team_name, zones):
    """Barras de la distribución de posiciones de un equipo, coloreadas por zona."""
    colors = [_ZONE_STYLE[z][1] if z in _ZONE_STYLE else "#94a3b8" for z in dist["zona"].fillna("")]
    fig = go.Figure(
        go.Bar(
            x=dist["posicion"],
            y=dist["prob"],
            marker_color=colors,
            hovertemplate="Puesto %{x}º: <b>%{y:.1%}</b><extra></extra>",
        )
    )
    fig.update_layout(
        title=f"¿Dónde acaba {team_name}?",
        xaxis={"title": "posición final", "dtick": 1},
        yaxis={"title": "probabilidad", "tickformat": ".0%"},
        height=340,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        bargap=0.15,
    )
    return fig


def page_projection(bundle, features, settings):
    st.header("Clasificación proyectada")
    if bundle is None:
        st.warning("No hay modelo entrenado. Ejecuta `alaves train`.")
        return
    season, from_md = _season_selector(features, settings)
    n = st.sidebar.select_slider("Simulaciones", [1000, 5000, 10000, 20000], value=10000)
    projection = project_standings(bundle, features, settings, season, from_md, n=n)
    if projection is None:
        st.info(
            f"No hay partidos por simular en {season}. El calendario de la temporada "
            "en curso se ingiere en la Fase 7."
        )
        return
    st.caption(
        f"{projection.n_played} jugados · {projection.n_remaining} por simular · "
        f"{n} simulaciones · variante {projection.variant}"
    )

    table = dd.projection_table(projection, settings)
    zones = settings.league.zones
    focus_name = dd.team_name(settings, settings.focus_team)
    tab_tabla, tab_mapa, tab_equipo = st.tabs(
        ["📋 Tabla", "🗺️ Mapa de posiciones", "🔍 Explorar equipo"]
    )

    with tab_tabla:
        _zone_legend(zones)
        st.caption(
            "El color viene del **puesto proyectado** (el orden de esta tabla), así que hay "
            "exactamente tantos equipos por zona como plazas reparte la liga. "
            "`Pos esperada` es la media de las simulaciones y puede diferir."
        )
        solo_zonas = st.checkbox("Mostrar solo zonas europeas y de descenso", value=False)
        shown = table[table["zona"].notna()] if solo_zonas else table
        display = shown.drop(columns=["team_id", "zona"])
        st.dataframe(
            _style_projection_table(display, shown["zona"]), width="stretch", hide_index=True
        )

    with tab_mapa:
        st.caption(
            "Cada fila es un equipo (líder arriba) y el color de la celda es la probabilidad "
            "de acabar en ese puesto. Las franjas verticales marcan las zonas: **verde** "
            f"Europa, **rojo** descenso. El rombo ámbar ◆ es la posición esperada y ★ marca "
            f"a {focus_name}."
        )
        show_pct = st.checkbox("Mostrar porcentajes en las celdas", value=True)
        heat = dd.position_heatmap(projection, settings)
        st.plotly_chart(
            _projection_heatmap(heat, table, focus_name, zones, show_pct=show_pct),
            width="stretch",
        )

    with tab_equipo:
        names = list(table["Equipo"])
        default = names.index(focus_name) if focus_name in names else 0
        chosen = st.selectbox("Equipo", names, index=default)
        team_id = table.loc[table["Equipo"] == chosen, "team_id"].iloc[0]
        row = table[table["Equipo"] == chosen].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Posición esperada", f"{row['Pos esperada']:.1f}º")
        c2.metric("Puntos esperados", f"{row['Pts esperados']:.1f}")
        zone = dd.zone_for_position(row["Pos esperada"], zones)
        c3.metric("Zona previsible", _ZONE_LABELS.get(zone, "Media tabla"))

        probs = dd.team_zone_probabilities(projection, team_id)
        cols = st.columns(len(probs))
        for col, (zone_key, value) in zip(cols, probs.items(), strict=True):
            col.metric(_ZONE_LABELS.get(zone_key, zone_key), f"{value * 100:.1f}%")

        dist = dd.team_position_distribution(projection, team_id)
        st.plotly_chart(_team_distribution_chart(dist, chosen, zones), width="stretch")


def _zone_legend(zones):
    """Leyenda de zonas con sus rangos de puestos.

    Los chips usan el MISMO pastel que las filas de la tabla (con un borde del
    color fuerte), para que leyenda y tabla se vean idénticas.
    """
    chips = []
    for key, (_short, color, pastel) in _ZONE_STYLE.items():
        rng = zones.get(key)
        if not rng:
            continue
        span = f"{rng[0]}º" if rng[0] == rng[1] else f"{rng[0]}º–{rng[1]}º"
        chips.append(
            f"<span style='background:{pastel};color:#111827;padding:3px 10px;"
            f"border-left:5px solid {color};border-radius:4px;margin-right:8px;"
            f"font-size:0.85em'>{_ZONE_LABELS[key]} · {span}</span>"
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def page_focus(bundle, features, settings):
    name = dd.team_name(settings, settings.focus_team)
    st.header(f"{name} en detalle")
    season, _ = _season_selector(features, settings)
    timeline = dd.focus_timeline(features, settings, season)
    if timeline.empty:
        st.info(f"{name} no tiene partidos en {season}.")
        return

    st.subheader("Elo por jornada")
    st.plotly_chart(
        px.line(timeline, x="matchday", y="elo", markers=True).update_layout(height=280),
        width="stretch",
    )
    st.subheader("xG a favor y en contra")
    fig = go.Figure()
    fig.add_scatter(
        x=timeline["matchday"], y=timeline["xg_favor"], name="xG a favor", mode="lines+markers"
    )
    fig.add_scatter(
        x=timeline["matchday"], y=timeline["xg_contra"], name="xG en contra", mode="lines+markers"
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Forma (puntos/partido, media 5)")
    st.plotly_chart(
        px.line(timeline, x="matchday", y="forma_pts_ma5", markers=True).update_layout(height=260),
        width="stretch",
    )


def page_explain(bundle, features, settings):
    st.header("Explicabilidad")
    if bundle is None:
        st.warning("No hay modelo entrenado. Ejecuta `alaves train`.")
        return
    model = bundle.variants[VARIANT_NO_ODDS].gbm
    finished = features[features["result"].notna()]
    sample = finished.sample(min(1000, len(finished)), random_state=42)

    st.subheader("Importancia global (SHAP) — variante sin cuotas")
    imp = importance.global_importance(model, sample).head(20)
    imp["significado"] = imp["feature"].map(lambda f: describe(f) or "")
    fig = px.bar(
        imp.iloc[::-1],
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        hover_data=["significado"],
    )
    fig.update_layout(height=520, xaxis_title="media |SHAP|", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Desglose de un partido (waterfall)")
    season, _ = _season_selector(features, settings)
    focus = settings.focus_team
    focus_rows = features[
        (features["season"] == season)
        & ((features["home_id"] == focus) | (features["away_id"] == focus))
    ]
    if focus_rows.empty:
        st.info("Sin partidos del equipo foco en esta temporada.")
        return
    labels = {
        r.match_id: (
            f"J{int(r.matchday)} — {dd.team_name(settings, r.home_id)} "
            f"vs {dd.team_name(settings, r.away_id)}"
        )
        for r in focus_rows.itertuples()
    }
    match_id = st.selectbox("Partido", list(labels), format_func=lambda m: labels[m])
    outcome = st.radio("Clase a explicar", ["H", "D", "A"], horizontal=True)
    row = focus_rows[focus_rows["match_id"] == match_id]
    contrib = importance.match_contributions(model, row, outcome).head(12)
    contrib["significado"] = contrib["feature"].map(lambda f: describe(f) or "")
    fig = go.Figure(
        go.Bar(
            x=contrib["shap"][::-1],
            y=contrib["feature"][::-1],
            orientation="h",
            marker_color=["#2ca02c" if v > 0 else "#d62728" for v in contrib["shap"][::-1]],
        )
    )
    fig.update_layout(height=420, xaxis_title=f"SHAP (→ empuja hacia {outcome})")
    st.plotly_chart(fig, width="stretch")


def page_performance(bundle, features, settings):
    st.header("Rendimiento del modelo")
    conn = db.connect(settings.data.db_path)
    try:
        registry = dd.model_registry_table(conn)
        log = dd.prediction_log(conn, settings)
    finally:
        conn.close()

    st.subheader("Registro de versiones")
    if registry.empty:
        st.info("Aún no hay modelos registrados. Ejecuta `alaves train`.")
    else:
        st.dataframe(registry, width="stretch", hide_index=True)

    st.subheader("Historial de predicciones")
    if log.empty:
        st.info("Aún no hay predicciones persistidas. Ejecuta `alaves predict`.")
        return
    resolved = log[log["result"].notna()]
    if not resolved.empty:
        acc = resolved["acierto"].mean()
        st.metric("Acierto en predicciones resueltas", f"{acc * 100:.1f}%", f"n={len(resolved)}")
    show = log[["Local", "Visitante", "pred_result", "result", "p_home", "p_draw", "p_away"]]
    st.dataframe(show, width="stretch", hide_index=True)


def page_decisions(settings):
    st.header("Registro de decisiones (ADRs)")
    st.caption("Transparencia total: cada decisión no trivial del proyecto está aquí.")
    adrs = dd.adr_list(Path("docs/decisions"))
    if adrs.empty:
        st.info("No se encontraron ADRs en docs/decisions/.")
        return
    labels = {r.path: f"ADR-{r.num:03d} — {r.titulo}" for r in adrs.itertuples()}
    choice = st.selectbox("ADR", list(labels), format_func=lambda p: labels[p])
    st.markdown(Path(choice).read_text(encoding="utf-8"))


def main():
    settings = _settings()
    bundle, features = _bundle_and_features()
    st.sidebar.title("⚽ Predictor Alavés")
    st.sidebar.caption("LaLiga 2026-27 · SPEC §9")
    pages = {
        "Próxima jornada": lambda: page_next_matchday(bundle, features, settings),
        "Clasificación proyectada": lambda: page_projection(bundle, features, settings),
        "El Alavés en detalle": lambda: page_focus(bundle, features, settings),
        "Explicabilidad": lambda: page_explain(bundle, features, settings),
        "Rendimiento del modelo": lambda: page_performance(bundle, features, settings),
        "Registro de decisiones": lambda: page_decisions(settings),
    }
    choice = st.sidebar.radio("Página", list(pages))
    pages[choice]()


# Streamlit ejecuta el script en cada interacción; se llama sin condición.
main()
