"""
PEG Scanner Model
=================
Analiza el PEG Ratio (Price/Earnings to Growth) de acciones del NYSE/NASDAQ.
Genera rankings de los PEG más bajos (potencialmente subvaloradas) y más altos.

El PEG Ratio = P/E Ratio / Tasa de crecimiento de ganancias esperada.
- PEG < 1  → la acción puede estar subvalorada respecto a su crecimiento.
- PEG > 1  → puede estar sobrevalorada.
- PEG negativo → ganancias negativas, se excluye del ranking.

Autor: Curso Python
Versión: 1.0
"""

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS

# Subconjunto de tickers con alta probabilidad de tener datos de PEG en yfinance
PEG_NYSE_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADP", "AIG", "AMD", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BKNG", "BRKB", "C", "CAT", "CL", "COST",
    "CRM", "CSCO", "CVS", "CVX", "D", "DD", "DE", "DHR", "DOW",
    "ECL", "EFX", "F", "FDX", "GE", "GILD", "GM", "GOOGL", "GS",
    "HAL", "HD", "HON", "HPQ", "IBM", "INTC", "ISRG", "JNJ", "JPM",
    "KMB", "KO", "LLY", "LMT", "MA", "MCD", "MDLZ", "MDT", "MELI",
    "META", "MMM", "MO", "MRK", "MSFT", "MU", "NEE", "NEM", "NFLX",
    "NKE", "NOW", "NVDA", "NVS", "ORCL", "OXY", "PEP", "PFE", "PG",
    "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCCO", "SLB", "SPGI",
    "SPOT", "T", "TGT", "TJX", "TMUS", "TSLA", "TSM", "UNH", "UNP",
    "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

TOP_N = 10


def _obtener_peg_bulk(tickers: list) -> pd.DataFrame:
    """
    Descarga el PEG Ratio para una lista de tickers usando yfinance.
    Retorna DataFrame con columnas: Ticker, Nombre, PE_Ratio, PEG_Ratio, Sector.
    Solo incluye tickers con PEG positivo (valores negativos o None se descartan).
    """
    resultados = []

    lote = 15
    for i in range(0, len(tickers), lote):
        sub = tickers[i : i + lote]
        for ticker_sym in sub:
            try:
                t = yf.Ticker(ticker_sym)
                info = t.info

                peg = info.get("pegRatio", None)
                pe  = info.get("trailingPE", None) or info.get("forwardPE", None)

                # Solo incluir PEG positivo y válido
                if peg is None or peg <= 0:
                    continue

                nombre  = info.get("shortName", ticker_sym)
                sector  = info.get("sector", "N/D")
                precio  = info.get("currentPrice") or info.get("regularMarketPrice")

                resultados.append({
                    "Ticker":    ticker_sym,
                    "Nombre":    nombre,
                    "Sector":    sector,
                    "P/E Ratio": round(pe, 2) if pe else None,
                    "PEG Ratio": round(peg, 2),
                    "Precio":    round(precio, 2) if precio else None,
                })

            except Exception:
                pass

    df = pd.DataFrame(resultados)
    if df.empty:
        return df

    return df.reset_index(drop=True)


class PEGScanner:
    """
    Escanea el PEG Ratio del mercado NYSE/NASDAQ y genera rankings.
    """

    def __init__(self):
        self.tickers = PEG_NYSE_TICKERS

    def escanear(self, progress_callback=None) -> dict:
        """
        Descarga PEG Ratios y retorna rankings.

        Returns
        -------
        dict {
            'bajo':  pd.DataFrame,   # PEG más bajos (potencial valor)
            'alto':  pd.DataFrame,   # PEG más altos (posible sobrevaluación)
            'todos': pd.DataFrame,   # todos los resultados
        }
        """
        if progress_callback:
            progress_callback(0.05)

        df = _obtener_peg_bulk(self.tickers)

        if progress_callback:
            progress_callback(0.95)

        cols = ["Ticker", "Nombre", "Sector", "P/E Ratio", "PEG Ratio", "Precio"]
        empty = pd.DataFrame(columns=cols)

        if df.empty:
            return {"bajo": empty, "alto": empty, "todos": empty}

        bajo  = df.sort_values("PEG Ratio", ascending=True).head(TOP_N).reset_index(drop=True)
        alto  = df.sort_values("PEG Ratio", ascending=False).head(TOP_N).reset_index(drop=True)
        todos = df.sort_values("PEG Ratio", ascending=True).reset_index(drop=True)

        if progress_callback:
            progress_callback(1.0)

        return {"bajo": bajo, "alto": alto, "todos": todos}
