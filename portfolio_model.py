"""
Portfolio Analyzer Model
========================
Lógica de análisis de portafolio de inversión usando datos reales
descargados desde Yahoo Finance con yfinance.

Autor: Curso Python
Versión: 1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────
# CATÁLOGO DE ACTIVOS DISPONIBLES
# ─────────────────────────────────────────────────────────────────

ACTIVOS = {
    # Acciones USA
    "Apple (AAPL)":         "AAPL",
    "Google (GOOGL)":       "GOOGL",
    "Tesla (TSLA)":         "TSLA",
    "Microsoft (MSFT)":     "MSFT",
    "Amazon (AMZN)":        "AMZN",
    "NVIDIA (NVDA)":        "NVDA",
    "Meta (META)":          "META",
    "JPMorgan (JPM)":       "JPM",
    # Latinoamérica
    "MercadoLibre (MELI)":  "MELI",
    "YPF (YPF)":            "YPF",
    "Globant (GLOB)":       "GLOB",
    "Banco Bradesco (BBD)": "BBD",
    "Vale (VALE)":          "VALE",
    "Embraer (ERJ)":        "ERJ",
}


class PortfolioAnalyzer:
    """
    Descarga precios históricos y calcula métricas de portafolio.
    """

    def __init__(self, tickers: list, periodo: str = "1y"):
        """
        Parameters
        ----------
        tickers : list  — lista de símbolos (ej: ['AAPL', 'MELI'])
        periodo : str   — período de datos ('6mo', '1y', '2y', '5y')
        """
        self.tickers = tickers
        self.periodo = periodo
        self.precios  = None
        self.retornos = None
        self._descargar()

    # ── Descarga de datos ────────────────────────────────────────
    def _descargar(self):
        """Descarga precios de cierre ajustados desde Yahoo Finance"""
        data = yf.download(
            self.tickers,
            period=self.periodo,
            auto_adjust=True,
            progress=False,
        )

        # Manejar tanto 1 ticker como múltiples
        if len(self.tickers) == 1:
            self.precios = data[["Close"]].rename(columns={"Close": self.tickers[0]})
        else:
            self.precios = data["Close"]

        # Eliminar columnas con todos NaN y filas con algún NaN
        self.precios  = self.precios.dropna(axis=1, how="all").dropna()
        self.retornos = self.precios.pct_change().dropna()

    # ── Métricas individuales ────────────────────────────────────
    def metricas_individuales(self) -> pd.DataFrame:
        """Retorno anualizado y volatilidad anualizada por activo"""
        retorno_anual = self.retornos.mean() * 252
        volatilidad   = self.retornos.std() * np.sqrt(252)
        sharpe        = retorno_anual / volatilidad

        df = pd.DataFrame({
            "Retorno Anual (%)":    (retorno_anual * 100).round(2),
            "Volatilidad Anual (%)": (volatilidad  * 100).round(2),
            "Sharpe Ratio":          sharpe.round(2),
        })
        return df

    # ── Métricas del portafolio ──────────────────────────────────
    def analizar_portafolio(self, pesos: dict) -> dict:
        """
        Calcula métricas del portafolio dado un diccionario de pesos.

        Parameters
        ----------
        pesos : dict — {ticker: peso_decimal}  (deben sumar 1)

        Returns
        -------
        dict con retorno, volatilidad, sharpe y evolución del portafolio
        """
        tickers_validos = [t for t in pesos if t in self.retornos.columns]
        w = np.array([pesos[t] for t in tickers_validos])

        # Normalizar por si no suman exactamente 1
        w = w / w.sum()

        ret = self.retornos[tickers_validos]

        # Retorno y volatilidad anualizados
        retorno_port = float((ret.mean() @ w) * 252)
        cov_matrix   = ret.cov() * 252
        volatilidad  = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe       = retorno_port / volatilidad if volatilidad > 0 else 0

        # Evolución acumulada del portafolio (base 100)
        retornos_port = (ret * w).sum(axis=1)
        evolucion     = (1 + retornos_port).cumprod() * 100

        # Máximo drawdown
        rolling_max  = evolucion.cummax()
        drawdown     = (evolucion - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min() * 100)

        return {
            "retorno_anual":   round(retorno_port * 100, 2),
            "volatilidad":     round(volatilidad  * 100, 2),
            "sharpe":          round(sharpe, 2),
            "max_drawdown":    round(max_drawdown, 2),
            "evolucion":       evolucion,
            "fechas":          evolucion.index,
        }

    # ── Correlación ──────────────────────────────────────────────
    def matriz_correlacion(self) -> pd.DataFrame:
        """Devuelve la matriz de correlación entre activos"""
        return self.retornos.corr().round(2)
