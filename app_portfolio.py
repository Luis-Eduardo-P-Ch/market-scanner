"""
Simulador de Portafolio de Inversión — Streamlit App
=====================================================
Interfaz para armar un portafolio con activos reales
y analizar retorno esperado, volatilidad y evolución histórica.
Incluye Scanner de Mercados: Top 10 mayores caídas y subidas en ByMA y NYSE/NASDAQ.

Autor: Curso Python
Versión: 3.0
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from portfolio_model import PortfolioAnalyzer, ACTIVOS
from market_scanner_model import MarketScanner, PERIODOS
from dividend_scanner_model import DividendScanner

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Portafolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f7f5f0;
        color: #1a1a1a;
    }
    h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        color: #1a1a1a;
        letter-spacing: -2px;
        font-size: 2.8rem;
    }
    h2, h3 { font-family: 'Syne', sans-serif; color: #1a1a1a; }

    .stButton>button {
        background-color: #1a1a1a;
        color: #f7f5f0;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 0px;
        padding: 0.75rem 2rem;
        width: 100%;
        font-size: 0.95rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .stButton>button:hover { background-color: #333; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e0ddd5;
        padding: 1.5rem;
        margin: 0.3rem 0;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
    }
    .positive { color: #1a7a4a; }
    .negative { color: #c0392b; }
    .neutral  { color: #1a1a1a; }
    .warning-text {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        margin-top: 0.5rem;
    }

    /* ── Scanner styles ── */
    .scanner-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
        border-bottom: 3px solid #1a1a1a;
        margin-bottom: 1rem;
    }
    .scanner-period-label {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.3rem;
    }
    .rank-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.75rem;
        border-bottom: 1px solid #eee;
        font-size: 0.88rem;
    }
    .rank-row:hover { background: #f0ede6; }
    .rank-num {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.75rem;
        color: #ccc;
        width: 1.5rem;
    }
    .rank-ticker {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        flex: 1;
        padding-left: 0.5rem;
    }
    .rank-change-loser  { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #c0392b; }
    .rank-change-gainer { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #1a7a4a; }
    .scanner-card {
        background: #ffffff;
        border: 1px solid #e0ddd5;
        padding: 1rem 1rem 0.5rem 1rem;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────

def _render_ranking_card(titulo: str, df_dict: dict, modo: str = "losers"):
    """Renderiza el bloque completo de un mercado con 4 períodos en columnas."""
    st.markdown(f'<div class="scanner-header">{titulo}</div>', unsafe_allow_html=True)

    periodos_nombres = list(PERIODOS.keys())
    cols = st.columns(4)
    change_class = "rank-change-loser" if modo == "losers" else "rank-change-gainer"

    for col, periodo in zip(cols, periodos_nombres):
        df = df_dict.get(periodo)
        with col:
            st.markdown(f'<div class="scanner-period-label">⏱ {periodo}</div>', unsafe_allow_html=True)
            if df is None or df.empty:
                st.caption("Sin datos disponibles.")
            else:
                rows_html = ""
                for i, row in df.iterrows():
                    ticker = str(row["Ticker"]).replace(".BA", "")
                    variacion = row["Variación (%)"]
                    rows_html += f"""
                    <div class="rank-row">
                        <span class="rank-num">{i+1}</span>
                        <span class="rank-ticker">{ticker}</span>
                        <span class="{change_class}">{variacion:+.1f}%</span>
                    </div>"""

                st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)


def _render_bar_chart(titulo: str, df_dict: dict, periodo_key: str, modo: str = "losers"):
    """Bar chart horizontal del ranking para un período dado."""
    df = df_dict.get(periodo_key)
    if df is None or df.empty:
        return

    df_plot = df.copy()
    df_plot["Ticker"] = df_plot["Ticker"].str.replace(".BA", "", regex=False)

    if modo == "losers":
        colors = [f"rgba(192,57,43,{0.4 + 0.06 * i})" for i in range(len(df_plot))]
    else:
        colors = [f"rgba(26,122,74,{0.4 + 0.06 * i})" for i in range(len(df_plot))]

    fig = go.Figure(go.Bar(
        x=df_plot["Variación (%)"],
        y=df_plot["Ticker"],
        orientation="h",
        marker_color=colors,
        text=df_plot["Variación (%)"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{titulo} — {periodo_key}",
        xaxis_title="Variación (%)",
        yaxis=dict(autorange="reversed"),
        height=400,
        template="plotly_white",
        margin=dict(l=10, r=60, t=40, b=30),
        xaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, width='stretch')


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Configurar Portafolio")
    st.markdown("---")

    nombres_activos = list(ACTIVOS.keys())
    seleccionados = st.multiselect(
        "Seleccioná los activos",
        options=nombres_activos,
        default=["Apple (AAPL)", "MercadoLibre (MELI)", "Tesla (TSLA)"],
    )

    periodo = st.selectbox(
        "Período histórico",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
        format_func=lambda x: {
            "6mo": "6 meses", "1y": "1 año", "2y": "2 años", "5y": "5 años"
        }[x],
    )

    st.markdown("---")

    pesos_input = {}
    if seleccionados:
        st.markdown("**Pesos del portafolio (%)**")
        peso_default = int(100 / len(seleccionados))
        total = 0
        for nombre in seleccionados:
            ticker = ACTIVOS[nombre]
            peso = st.slider(
                f"{nombre.split('(')[0].strip()}",
                min_value=0, max_value=100,
                value=peso_default, step=5,
                key=f"peso_{ticker}"
            )
            pesos_input[ticker] = peso
            total += peso

        if total != 100:
            st.warning(f"Los pesos suman {total}%. Se normalizarán automáticamente a 100%.")
        else:
            st.success("✓ Los pesos suman 100%")

    st.markdown("---")
    analizar = st.button("▶  ANALIZAR PORTAFOLIO")

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("# Simulador de Portafolio")
st.markdown("Datos reales desde Yahoo Finance · Curso Python")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab_portfolio, tab_scanner, tab_dividendos = st.tabs(["📈 Portafolio", "📉 Scanner de Mercados", "💰 Ranking de Dividendos"])

# ══════════════════════════════════════════════
# TAB 1 — PORTAFOLIO (lógica original intacta)
# ══════════════════════════════════════════════
with tab_portfolio:

    if analizar:
        if not seleccionados:
            st.warning("Seleccioná al menos un activo para continuar.")
        elif sum(pesos_input.values()) == 0:
            st.warning("Asigná pesos mayores a 0 para continuar.")
        else:
            tickers   = [ACTIVOS[n] for n in seleccionados]
            pesos_dec = {t: p / 100 for t, p in pesos_input.items()}

            with st.spinner("Descargando datos desde Yahoo Finance..."):
                try:
                    analyzer  = PortfolioAnalyzer(tickers, periodo)
                    resultado = analyzer.analizar_portafolio(pesos_dec)
                    metricas  = analyzer.metricas_individuales()
                    st.session_state.analyzer      = analyzer
                    st.session_state.resultado     = resultado
                    st.session_state.metricas      = metricas
                    st.session_state.seleccionados = seleccionados
                    st.session_state.pesos_input   = pesos_input
                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")
                    st.stop()

    if "resultado" in st.session_state:
        r        = st.session_state.resultado
        ana      = st.session_state.analyzer
        metricas = st.session_state.metricas

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        ret_color  = "positive" if r["retorno_anual"] >= 0 else "negative"
        draw_color = "negative" if r["max_drawdown"]  < -15 else "neutral"

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Retorno Anual Est.</p>
                <p class="metric-value {ret_color}">{r['retorno_anual']:+.1f}%</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Volatilidad Anual</p>
                <p class="metric-value neutral">{r['volatilidad']:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Sharpe Ratio</p>
                <p class="metric-value neutral">{r['sharpe']:.2f}</p>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Máx. Drawdown</p>
                <p class="metric-value {draw_color}">{r['max_drawdown']:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Evolución", "📊 Activos Individuales", "🔗 Correlaciones"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=r["fechas"], y=r["evolucion"],
                mode="lines", name="Mi Portafolio",
                line=dict(color="#1a1a1a", width=3),
                fill="tozeroy", fillcolor="rgba(26,26,26,0.05)",
            ))
            fig.add_hline(y=100, line_dash="dot", line_color="#aaa",
                          annotation_text="Base 100", annotation_position="right")
            fig.update_layout(
                title="Evolución del Portafolio (Base 100)",
                xaxis_title="Fecha", yaxis_title="Valor (Base 100)",
                height=450, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown('<p class="warning-text">* Retorno histórico no garantiza rendimiento futuro. Solo con fines educativos.</p>',
                        unsafe_allow_html=True)

        with tab2:
            col_a, col_b = st.columns([2, 1])

            with col_a:
                df_scatter = metricas.reset_index()
                df_scatter = df_scatter.rename(columns={df_scatter.columns[0]: "Ticker"})
                fig2 = px.scatter(
                    df_scatter,
                    x="Volatilidad Anual (%)", y="Retorno Anual (%)",
                    text="Ticker",
                    size=[20] * len(df_scatter),
                    color="Sharpe Ratio", color_continuous_scale="RdYlGn",
                    title="Retorno vs Riesgo por Activo",
                )
                fig2.update_traces(textposition="top center", marker=dict(sizemode="diameter"))
                fig2.add_hline(y=0, line_dash="dot", line_color="#ccc")
                fig2.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig2, width='stretch')

            with col_b:
                st.markdown("### Tabla de Métricas")
                st.dataframe(
                    metricas.style.format({
                        "Retorno Anual (%)":     "{:+.2f}%",
                        "Volatilidad Anual (%)": "{:.2f}%",
                        "Sharpe Ratio":          "{:.2f}",
                    }),
                    width='stretch',
                )

            pesos_norm  = st.session_state.pesos_input
            total_pesos = sum(pesos_norm.values())
            labels = [n.split("(")[0].strip() for n in st.session_state.seleccionados]
            values = [pesos_norm[ACTIVOS[n]] / total_pesos * 100 for n in st.session_state.seleccionados]

            fig3 = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.45,
                marker=dict(colors=px.colors.qualitative.G10),
            ))
            fig3.update_layout(title="Composición del Portafolio", height=350, template="plotly_white")
            st.plotly_chart(fig3, width='stretch')

        with tab3:
            if len(ana.retornos.columns) > 1:
                corr = ana.matriz_correlacion()
                fig4 = px.imshow(
                    corr, text_auto=True,
                    color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                    title="Matriz de Correlación entre Activos", aspect="auto",
                )
                fig4.update_layout(height=450, template="plotly_white")
                st.plotly_chart(fig4, width='stretch')
                st.info("💡 Correlaciones cercanas a -1 indican activos que se mueven en sentidos opuestos, lo que **reduce el riesgo** del portafolio.")
            else:
                st.info("Seleccioná más de un activo para ver correlaciones.")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#bbb;">
            <p style="font-size:5rem; margin:0;">📈</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.2rem; color:#aaa; margin-top:1rem;">
                Elegí los activos y los pesos en el panel izquierdo<br>
                y hacé clic en <strong style="color:#1a1a1a;">ANALIZAR PORTAFOLIO</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SCANNER DE MERCADOS
# ══════════════════════════════════════════════
with tab_scanner:
    st.markdown("### 🔍 Scanner de Mercados")
    st.markdown("Analizá las acciones con mejor y peor rendimiento en los últimos 1, 3, 6 y 12 meses.")
    st.markdown("---")

    # ── Fila de controles ────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

    with ctrl1:
        mercado_sel = st.radio(
            "Mercado:",
            options=["Mercado Argentino (ByMA)", "Mercado de Nueva York (NYSE/NASDAQ)", "Ambos mercados"],
            horizontal=True,
            key="scanner_mercado",
        )

    with ctrl2:
        # ── NUEVO: selector losers / gainers ─────────────────────
        modo_sel = st.radio(
            "Tipo de ranking:",
            options=["losers", "gainers"],
            format_func=lambda x: "📉 Mayores caídas" if x == "losers" else "📈 Mayores subidas",
            horizontal=False,
            key="scanner_modo",
        )

    with ctrl3:
        vista_sel = st.selectbox(
            "Vista",
            options=["Ranking tablas", "Gráfico de barras"],
            key="scanner_vista",
        )

    if vista_sel == "Gráfico de barras":
        periodo_grafico = st.selectbox(
            "Período para el gráfico",
            options=list(PERIODOS.keys()),
            key="scanner_periodo_grafico",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Título dinámico según modo ───────────────────────────────
    titulo_boton = "🔍  ESCANEAR MERCADO"
    escanear_btn = st.button(titulo_boton, key="btn_escanear")

    if escanear_btn:
        mercados_a_correr = []
        if mercado_sel == "Mercado Argentino (ByMA)":
            mercados_a_correr = [("byma", "🇦🇷 Mercado Argentino (ByMA)")]
        elif mercado_sel == "Mercado de Nueva York (NYSE/NASDAQ)":
            mercados_a_correr = [("nyse", "🇺🇸 NYSE / NASDAQ")]
        else:
            mercados_a_correr = [
                ("byma", "🇦🇷 Mercado Argentino (ByMA)"),
                ("nyse", "🇺🇸 NYSE / NASDAQ"),
            ]

        resultados_scanner = {}

        for mercado_key, mercado_label in mercados_a_correr:
            with st.spinner(f"Descargando datos de {mercado_label}... (puede tardar ~30 seg)"):
                try:
                    scanner = MarketScanner(mercado=mercado_key)
                    prog_bar = st.progress(0, text=f"Analizando {mercado_label}...")

                    def _prog(val, label=mercado_label, bar=prog_bar):
                        bar.progress(val, text=f"Procesando {label}... {int(val*100)}%")

                    # ── Se pasa el modo al modelo ─────────────────
                    resultados = scanner.escanear_todos_periodos(
                        modo=modo_sel,
                        progress_callback=_prog,
                    )
                    prog_bar.empty()
                    resultados_scanner[mercado_key] = (mercado_label, resultados)

                except Exception as e:
                    st.error(f"Error al escanear {mercado_label}: {e}")

        if resultados_scanner:
            st.session_state.scanner_resultados  = resultados_scanner
            st.session_state.scanner_vista_val   = vista_sel
            st.session_state.scanner_modo_val    = modo_sel
            if vista_sel == "Gráfico de barras":
                st.session_state.scanner_periodo_val = periodo_grafico

    # ── Mostrar resultados almacenados ───────────────────────────
    if "scanner_resultados" in st.session_state:
        resultados_scanner = st.session_state.scanner_resultados
        vista              = st.session_state.get("scanner_vista_val", "Ranking tablas")
        modo_guardado      = st.session_state.get("scanner_modo_val", "losers")

        # Badge de modo actual
        if modo_guardado == "gainers":
            st.success("📈 Mostrando: **Top 10 Mayores Subidas**")
        else:
            st.error("📉 Mostrando: **Top 10 Mayores Caídas**")

        for mercado_key, (mercado_label, df_dict) in resultados_scanner.items():

            if vista == "Ranking tablas":
                _render_ranking_card(mercado_label, df_dict, modo=modo_guardado)

            else:
                periodo_g = st.session_state.get("scanner_periodo_val", "1 mes")
                _render_bar_chart(mercado_label, df_dict, periodo_g, modo=modo_guardado)

            # Tabla descargable por período
            with st.expander(f"📋 Ver datos completos — {mercado_label}"):
                tab_cols = st.tabs(list(PERIODOS.keys()))
                for tab_col, periodo_nombre in zip(tab_cols, PERIODOS.keys()):
                    with tab_col:
                        df = df_dict.get(periodo_nombre)
                        if df is not None and not df.empty:
                            df_display = df.copy()
                            df_display["Ticker"] = df_display["Ticker"].str.replace(".BA", "", regex=False)
                            st.dataframe(
                                df_display.style.format({"Variación (%)": "{:+.2f}%"}),
                                width='stretch',
                                hide_index=True,
                            )
                        else:
                            st.caption("Sin datos para este período.")

            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="warning-text">* Los datos provienen de Yahoo Finance. Algunos símbolos de ByMA pueden no estar disponibles. Solo con fines educativos.</p>',
                    unsafe_allow_html=True)

    elif not escanear_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">🔍</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado, el tipo de ranking y hacé clic en<br>
                <strong style="color:#1a1a1a;">ESCANEAR MERCADO</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — RANKING DE DIVIDENDOS
# ══════════════════════════════════════════════
with tab_dividendos:
    st.markdown("### 💰 Ranking de Dividendos")
    st.markdown("Compará qué acciones pagan más y menos dividendos según el **Dividend Yield** anual.")
    st.markdown("---")

    # ── Controles ──────────────────────────────────────────────────
    div_ctrl1, div_ctrl2, div_ctrl3 = st.columns([2, 1, 1])

    with div_ctrl1:
        div_mercado = st.radio(
            "Mercado:",
            options=["NYSE / NASDAQ", "Mercado Argentino (ByMA)"],
            horizontal=True,
            key="div_mercado",
        )

    with div_ctrl2:
        div_periodo_dias = st.selectbox(
            "Período de dividendos:",
            options=[365, 180, 90],
            format_func=lambda x: {365: "Últimos 12 meses", 180: "Últimos 6 meses", 90: "Últimos 3 meses"}[x],
            key="div_periodo",
        )

    with div_ctrl3:
        div_vista = st.selectbox(
            "Vista:",
            options=["Tablas", "Gráfico de barras"],
            key="div_vista",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    div_btn = st.button("💰  ESCANEAR DIVIDENDOS", key="btn_dividendos")

    if div_btn:
        mercado_key = "byma" if "ByMA" in div_mercado else "nyse"
        with st.spinner("Descargando datos de dividendos... (puede tardar ~1 min)"):
            try:
                scanner_div = DividendScanner(mercado=mercado_key)
                prog_div = st.progress(0, text="Analizando dividendos...")

                def _prog_div(val):
                    prog_div.progress(val, text=f"Analizando dividendos... {int(val*100)}%")

                resultados_div = scanner_div.escanear(dias=div_periodo_dias, progress_callback=_prog_div)
                prog_div.empty()

                st.session_state.div_resultados  = resultados_div
                st.session_state.div_mercado_lbl = div_mercado
                st.session_state.div_vista_val   = div_vista

            except Exception as e:
                st.error(f"Error al escanear dividendos: {e}")

    # ── Mostrar resultados ──────────────────────────────────────────
    if "div_resultados" in st.session_state:
        res      = st.session_state.div_resultados
        lbl      = st.session_state.get("div_mercado_lbl", "")
        vista_d  = st.session_state.get("div_vista_val", "Tablas")

        df_top    = res.get("top")
        df_bottom = res.get("bottom")
        df_todos  = res.get("todos")

        if df_top is None or df_top.empty:
            st.warning("No se encontraron datos de dividendos para este mercado y período.")
        else:
            if vista_d == "Tablas":
                col_top, col_bot = st.columns(2)

                # ── TOP PAYERS ──────────────────────────────────────
                with col_top:
                    st.markdown(
                        '<div class="scanner-header" style="color:#1a7a4a;">📈 Mayores Pagadores de Dividendos</div>',
                        unsafe_allow_html=True,
                    )
                    rows_html = ""
                    for i, row in df_top.iterrows():
                        ticker = str(row["Ticker"]).replace(".BA", "")
                        nombre = str(row.get("Nombre", ticker))[:28]
                        yld    = row["Dividend_Yield (%)"]
                        rows_html += f"""
                        <div class="rank-row">
                            <span class="rank-num">{i+1}</span>
                            <span class="rank-ticker">{ticker}</span>
                            <span style="font-size:0.75rem; color:#888; flex:2; padding: 0 0.5rem;">{nombre}</span>
                            <span class="rank-change-gainer">{yld:.2f}%</span>
                        </div>"""
                    st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

                # ── BOTTOM PAYERS ───────────────────────────────────
                with col_bot:
                    st.markdown(
                        '<div class="scanner-header" style="color:#c0392b;">📉 Menores Pagadores de Dividendos</div>',
                        unsafe_allow_html=True,
                    )
                    if df_bottom is None or df_bottom.empty:
                        st.caption("Sin datos suficientes.")
                    else:
                        rows_html = ""
                        for i, row in df_bottom.iterrows():
                            ticker = str(row["Ticker"]).replace(".BA", "")
                            nombre = str(row.get("Nombre", ticker))[:28]
                            yld    = row["Dividend_Yield (%)"]
                            rows_html += f"""
                            <div class="rank-row">
                                <span class="rank-num">{i+1}</span>
                                <span class="rank-ticker">{ticker}</span>
                                <span style="font-size:0.75rem; color:#888; flex:2; padding: 0 0.5rem;">{nombre}</span>
                                <span class="rank-change-loser">{yld:.2f}%</span>
                            </div>"""
                        st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

            else:
                # ── VISTA GRÁFICO ───────────────────────────────────
                import plotly.graph_objects as go

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    if df_top is not None and not df_top.empty:
                        df_plot = df_top.copy()
                        df_plot["Ticker"] = df_plot["Ticker"].str.replace(".BA", "", regex=False)
                        colors = [f"rgba(26,122,74,{0.4 + 0.055*i})" for i in range(len(df_plot))]
                        fig_top = go.Figure(go.Bar(
                            x=df_plot["Dividend_Yield (%)"],
                            y=df_plot["Ticker"],
                            orientation="h",
                            marker_color=colors,
                            text=df_plot["Dividend_Yield (%)"].apply(lambda v: f"{v:.2f}%"),
                            textposition="outside",
                        ))
                        fig_top.update_layout(
                            title="📈 Mayores Dividend Yield",
                            xaxis_title="Dividend Yield (%)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            template="plotly_white",
                            margin=dict(l=10, r=70, t=40, b=30),
                            xaxis=dict(ticksuffix="%"),
                        )
                        st.plotly_chart(fig_top, width="stretch")

                with col_g2:
                    if df_bottom is not None and not df_bottom.empty:
                        df_plot2 = df_bottom.copy()
                        df_plot2["Ticker"] = df_plot2["Ticker"].str.replace(".BA", "", regex=False)
                        colors2 = [f"rgba(192,57,43,{0.35 + 0.065*i})" for i in range(len(df_plot2))]
                        fig_bot = go.Figure(go.Bar(
                            x=df_plot2["Dividend_Yield (%)"],
                            y=df_plot2["Ticker"],
                            orientation="h",
                            marker_color=colors2,
                            text=df_plot2["Dividend_Yield (%)"].apply(lambda v: f"{v:.2f}%"),
                            textposition="outside",
                        ))
                        fig_bot.update_layout(
                            title="📉 Menores Dividend Yield (pero positivos)",
                            xaxis_title="Dividend Yield (%)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            template="plotly_white",
                            margin=dict(l=10, r=70, t=40, b=30),
                            xaxis=dict(ticksuffix="%"),
                        )
                        st.plotly_chart(fig_bot, width="stretch")

            # ── Tabla completa descargable ──────────────────────────
            with st.expander("📋 Ver tabla completa de dividendos"):
                if df_todos is not None and not df_todos.empty:
                    df_show = df_todos.copy()
                    df_show["Ticker"] = df_show["Ticker"].str.replace(".BA", "", regex=False)
                    st.dataframe(
                        df_show.style.format({
                            "Dividend_Yield (%)": "{:.2f}%",
                            "Dividendo_Total":    "{:.4f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

            st.markdown(
                '<p class="warning-text">* Dividend Yield = dividendos anuales / precio actual. '
                'Datos de Yahoo Finance. Solo con fines educativos.</p>',
                unsafe_allow_html=True,
            )

    elif not div_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">💰</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado y hacé clic en<br>
                <strong style="color:#1a1a1a;">ESCANEAR DIVIDENDOS</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#aaa; font-size:0.75rem;">
    Simulador de Portafolio · Datos: Yahoo Finance · Solo con fines educativos
</p>
""", unsafe_allow_html=True)
