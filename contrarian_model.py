"""
Contrarian Multifactor Model
=============================
Identifica acciones que más cayeron en precio en un período dado
y las ranquea usando un modelo multifactor (Momentum + Valoración + Calidad + Dividendos).

Lógica "contrarian":
  1. Se toman las N acciones con mayor caída de precio en el período seleccionado.
  2. A ese subconjunto se le aplica el mismo scoring multifactor del Smart Ranking.
  3. El resultado muestra las "caídas con mejores fundamentos": candidatas a rebote.

Factores y pesos:
  - Momentum  (6m + 12m promedio) : 40%
  - Valoración (PER inverso)       : 20%
  - Calidad   (ROA + Margen + ROE) : 30%
  - Dividendos (Dividend Yield)    : 10%

Autor: Curso Python
Versión: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS, _descargar_precios, _calcular_variacion

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

PESOS_C = {
    "momentum":   0.40,
    "valoracion": 0.20,
    "calidad":    0.30,
    "dividendos": 0.10,
}

# Cuántas acciones "más caídas" tomar como universo contrarian
UNIVERSO_CAIDAS = 40

# Top N final a mostrar
TOP_N = 20

PERIODOS_CONTRARIAN = {
    "1 mes":    30,
    "3 meses":  90,
    "6 meses":  180,
    "12 meses": 365,
}


# ─────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────

def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(method="average", ascending=ascending, na_option="keep")
    n      = series.notna().sum()
    pct    = (ranked - 1) / (n - 1) * 100 if n > 1 else ranked * 0
    return pct.round(2)


def _calcular_momentum(tickers: list) -> pd.DataFrame:
    """Calcula momentum 6m y 12m para una lista de tickers."""
    end   = datetime.today()
    start = end - timedelta(days=400)

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            return pd.DataFrame()

        precios = data["Close"] if len(tickers) > 1 else data[["Close"]].rename(columns={"Close": tickers[0]})
        precios = precios.dropna(axis=1, how="all")

    except Exception:
        return pd.DataFrame()

    resultados = []
    precio_actual = precios.iloc[-1]

    for sym in precios.columns:
        serie = precios[sym].dropna()
        if len(serie) < 60:
            continue

        fecha_6m  = serie.index[-1] - timedelta(days=180)
        hist_6m   = serie[serie.index <= fecha_6m]
        fecha_12m = serie.index[-1] - timedelta(days=365)
        hist_12m  = serie[serie.index <= fecha_12m]

        p_actual = float(precio_actual[sym])
        mom_6m   = None
        mom_12m  = None

        if not hist_6m.empty:
            mom_6m  = (p_actual - float(hist_6m.iloc[-1])) / float(hist_6m.iloc[-1]) * 100
        if not hist_12m.empty:
            mom_12m = (p_actual - float(hist_12m.iloc[-1])) / float(hist_12m.iloc[-1]) * 100

        if mom_6m is not None and mom_12m is not None:
            resultados.append({
                "Ticker": sym,
                "mom_6m":  round(mom_6m,  2),
                "mom_12m": round(mom_12m, 2),
                "mom_avg": round((mom_6m + mom_12m) / 2, 2),
            })

    return pd.DataFrame(resultados)


def _calcular_fundamentals(tickers: list, progress_callback=None) -> pd.DataFrame:
    """Descarga PER, ROA, Margen, ROE y Dividend Yield."""
    resultados = []
    total = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            t    = yf.Ticker(sym)
            info = t.info or {}

            nombre = info.get("shortName", sym)

            per = info.get("trailingPE", None)
            if per is not None:
                per = float(per)
                if per <= 0 or per > 1000:
                    per = None

            roa    = info.get("returnOnAssets",  None)
            roe    = info.get("returnOnEquity",  None)
            margen = info.get("profitMargins",   None)

            roa_pct    = round(roa    * 100, 2) if roa    is not None else None
            roe_pct    = round(roe    * 100, 2) if roe    is not None else None
            margen_pct = round(margen * 100, 2) if margen is not None else None

            dy = info.get("dividendYield", None)
            dy_pct = round(dy * 100, 2) if dy is not None else 0.0

            resultados.append({
                "Ticker":     sym,
                "Nombre":     nombre,
                "PER":        per,
                "ROA_pct":    roa_pct,
                "ROE_pct":    roe_pct,
                "Margen_pct": margen_pct,
                "DY_pct":     dy_pct,
            })

        except Exception:
            pass

        if progress_callback:
            progress_callback((i + 1) / total)

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class ContrarianScanner:
    """
    Modelo Contrarian Multifactor:
      1. Filtra las acciones con mayor caída de precio en el período indicado.
      2. Aplica scoring multifactor a ese universo.
      3. Retorna un ranking de "caídas con mejores fundamentos".
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_TICKERS if mercado == "byma" else NYSE_TICKERS

    def calcular(self, dias: int = 180, progress_callback=None) -> pd.DataFrame:
        """
        Ejecuta el modelo completo.

        Parameters
        ----------
        dias : int — período de caída a analizar (30, 90, 180, 365)
        progress_callback : callable(float, str) opcional

        Returns
        -------
        pd.DataFrame ordenado por smart_score descendente, con columnas:
            Ticker, Nombre, caida_periodo, smart_score,
            score_momentum, score_valoracion, score_calidad, score_dividendos,
            mom_6m, mom_12m, PER, ROA_pct, ROE_pct, Margen_pct, DY_pct
        """

        # ── 1. Descargar precios y calcular caída del período ────
        if progress_callback:
            progress_callback(0.05, "Descargando precios del mercado...")

        precios = _descargar_precios(self.tickers, dias)
        if precios.empty:
            return pd.DataFrame()

        variacion = _calcular_variacion(precios, dias)
        if variacion.empty:
            return pd.DataFrame()

        # ── 2. Tomar las N acciones con mayor caída ──────────────
        caidas = variacion.sort_values().head(UNIVERSO_CAIDAS)
        tickers_caidos = caidas.index.tolist()

        if progress_callback:
            progress_callback(0.20, f"Analizando las {len(tickers_caidos)} acciones más caídas...")

        # DataFrame de caídas
        df_caidas = pd.DataFrame({
            "Ticker":          tickers_caidos,
            "caida_periodo":   caidas.values.round(2),
        })

        # ── 3. Momentum ──────────────────────────────────────────
        if progress_callback:
            progress_callback(0.30, "Calculando momentum (6m / 12m)...")

        df_mom = _calcular_momentum(tickers_caidos)

        # ── 4. Fundamentals ──────────────────────────────────────
        def _prog_fund(val):
            if progress_callback:
                pct = 0.35 + val * 0.55
                progress_callback(pct, f"Descargando fundamentals... {int(val*100)}%")

        df_fund = _calcular_fundamentals(tickers_caidos, progress_callback=_prog_fund)

        if progress_callback:
            progress_callback(0.92, "Calculando scores multifactor...")

        # ── 5. Merge ─────────────────────────────────────────────
        if df_mom.empty or df_fund.empty:
            return pd.DataFrame()

        df = pd.merge(df_caidas, df_mom,  on="Ticker", how="inner")
        df = pd.merge(df,        df_fund, on="Ticker", how="inner")

        if df.empty:
            return pd.DataFrame()

        # ── 6. Scores por factor (percentil rank 0–100) ──────────

        # Momentum: mayor retorno → mejor
        df["score_momentum"] = _percentile_rank(df["mom_avg"], ascending=True)

        # Valoración: menor PER → mejor
        df["score_valoracion"] = _percentile_rank(df["PER"], ascending=False).fillna(50)

        # Calidad: promedio de ROA, ROE y Margen
        scores_calidad = []
        for col in ["ROA_pct", "ROE_pct", "Margen_pct"]:
            s = _percentile_rank(df[col], ascending=True).fillna(50)
            scores_calidad.append(s)
        df["score_calidad"] = pd.concat(scores_calidad, axis=1).mean(axis=1).round(2)

        # Dividendos: mayor yield → mejor
        df["score_dividendos"] = _percentile_rank(df["DY_pct"], ascending=True).fillna(0)

        # ── 7. Smart Score ponderado ─────────────────────────────
        df["smart_score"] = (
            df["score_momentum"]   * PESOS_C["momentum"]   +
            df["score_valoracion"] * PESOS_C["valoracion"] +
            df["score_calidad"]    * PESOS_C["calidad"]    +
            df["score_dividendos"] * PESOS_C["dividendos"]
        ).round(1)

        # ── 8. Ordenar por smart_score y devolver Top N ──────────
        df = df.sort_values("smart_score", ascending=False).head(TOP_N).reset_index(drop=True)
        df.index = df.index + 1

        if progress_callback:
            progress_callback(1.0, "¡Listo!")

        cols = [
            "Ticker", "Nombre", "caida_periodo",
            "smart_score",
            "score_momentum", "score_valoracion", "score_calidad", "score_dividendos",
            "mom_6m", "mom_12m",
            "PER", "ROA_pct", "ROE_pct", "Margen_pct", "DY_pct",
        ]
        return df[[c for c in cols if c in df.columns]]
