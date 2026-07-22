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
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS
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
    matchday = st.selectbox("Jornada", mds, index=0)
    variant = VARIANT_NO_ODDS

    def predict(rows):
        return bundle.predict_matches(rows, variant)

    preds = dd.matchday_predictions(predict, features, settings, season, matchday)
    if preds.empty:
        st.info("No hay partidos para esa jornada.")
        return

    focus = settings.focus_team
    focus_pred = preds[(preds["home_id"] == focus) | (preds["away_id"] == focus)]
    if not focus_pred.empty:
        p = focus_pred.iloc[0]
        st.subheader(f"{p['Local']} vs {p['Visitante']} — Jornada {matchday}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P(victoria local)", f"{p['p_home'] * 100:.1f}%")
        c2.metric("P(empate)", f"{p['p_draw'] * 100:.1f}%")
        c3.metric("P(victoria visitante)", f"{p['p_away'] * 100:.1f}%")
        c4.metric("Marcador más probable", p["pred_score"])
        fig = go.Figure(
            go.Bar(
                x=["Local", "Empate", "Visitante"],
                y=[p["p_home"], p["p_draw"], p["p_away"]],
                marker_color=["#2ca02c", "#7f7f7f", "#d62728"],
            )
        )
        fig.update_layout(yaxis_tickformat=".0%", height=280, title="Distribución 1X2")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Resto de la jornada")
    cols = ["Local", "Visitante", "Predicho", "p_home", "p_draw", "p_away", "pred_score", "Real"]
    table = preds[cols].rename(
        columns={"p_home": "P(1)", "p_draw": "P(X)", "p_away": "P(2)", "pred_score": "Marcador"}
    )
    for c in ("P(1)", "P(X)", "P(2)"):
        table[c] = _pct(table[c])
    st.dataframe(table, width="stretch", hide_index=True)


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
    display = table.drop(columns="team_id").copy()
    for c in ("P(título)", "P(Champions)", "P(Europa)", "P(descenso)"):
        display[c] = _pct(display[c])
    st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Distribución de posiciones")
    heat = dd.position_heatmap(projection, settings)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues", labels={"color": "P"})
    fig.update_layout(height=max(300, 22 * len(heat)), xaxis_title="posición final")
    st.plotly_chart(fig, width="stretch")


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
