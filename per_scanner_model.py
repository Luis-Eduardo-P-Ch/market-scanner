"""
PER Scanner Model
=================
Analiza el P/E Ratio (Price to Earnings) de acciones del NYSE/NASDAQ y ByMA.
Genera rankings de PER más bajo (potencialmente baratas) y PER más alto (caras).

- Trailing P/E  → basado en ganancias reales de los últimos 12 meses
- Forward P/E   → basado en ganancias estimadas para los próximos 12 meses

Un PER bajo puede indicar que la acción está subvalorada.
Un PER alto puede reflejar altas expectativas de crecimiento o sobrevaluación.
Los PER negativos (pérdidas) se excluyen del ranking.

Autor: Curso Python
Versión: 1.0
"""

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# ─────────────────────────────────────────────────────────────────
# LISTAS DE TICKERS
# ─────────────────────────────────────────────────────────────────

NYSE_PER_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADP", "AIG", "AMD", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BKNG", "BRKB", "C", "CAT", "CL", "COIN",
    "COST", "CRM", "CSCO", "CVS", "CVX", "DD", "DE", "DHR", "DOW",
    "ECL", "EFX", "F", "FDX", "GE", "GILD", "GM", "GOOGL", "GS",
    "HAL", "HD", "HON", "HPQ", "IBM", "INTC", "ISRG", "JNJ", "JPM",
    "KMB", "KO", "LLY", "LMT", "MA", "MCD", "MDLZ", "MDT", "MELI",
    "META", "MMM", "MO", "MRK", "MSFT", "MU", "NEE", "NEM", "NFLX",
    "NKE", "NOW", "NVDA", "NVS", "ORCL", "OXY", "PANW", "PEP",
    "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX", "SBUX",
    "SCCO", "SLB", "SNOW", "SPGI", "SPOT", "T", "TGT", "TJX",
    "TMUS", "TSLA", "TSM", "UBER", "UNH", "UNP", "USB", "V",
    "VALE", "VZ", "WFC", "WMT", "XOM",
]

BYMA_PER_TICKERS = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "CEPU.BA",
    "ALUA.BA", "TXAR.BA", "COME.BA", "TGSU2.BA", "EDN.BA",
    "BBAR.BA", "CRES.BA", "MIRG.BA", "SUPV.BA", "LOMA.BA",
    "CVH.BA", "IRSA.BA", "METR.BA", "TECO2.BA", "VALO.BA",
    "BYMA.BA", "AGRO.BA", "RICH.BA", "SEMI.BA", "MOLI.BA",
    "ROSE.BA", "CARC.BA", "INVJ.BA", "GARO.BA", "HARG.BA",
    "BOLT.BA", "CAPX.BA", "CELU.BA", "DGCU2.BA", "FERR.BA",
    "GCDI.BA", "LONG.BA", "MORI.BA", "SAMI.BA", "TRAN.BA",
]

TOP_N = 10


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN DE DESCARGA
# ─────────────────────────────────────────────────────────────────

def _obtener_per_bulk(tickers: list) -> pd.DataFrame:
    """
    Descarga Trailing P/E y Forward P/E para una lista de tickers.
    Retorna DataFrame con columnas:
        Ticker, Nombre, Sector, Precio, Trailing P/E, Forward P/E
    Solo incluye tickers con al menos un P/E positivo disponible.
    """
    resultados = []

    lote = 20
    for i in range(0, len(tickers), lote):
        sub = tickers[i : i + lote]
        for ticker_sym in sub:
            try:
                info = yf.Ticker(ticker_sym).info

                trailing = info.get("trailingPE", None)
                forward  = info.get("forwardPE",  None)

                # Descartar si ambos son negativos o nulos
                trailing = trailing if (trailing and trailing > 0) else None
                forward  = forward  if (forward  and forward  > 0) else None

                if trailing is None and forward is None:
                    continue

                nombre = info.get("shortName", ticker_sym)
                sector = info.get("sector", "N/D")
                precio = info.get("currentPrice") or info.get("regularMarketPrice")

                resultados.append({
                    "Ticker":      ticker_sym,
                    "Nombre":      nombre,
                    "Sector":      sector,
                    "Precio":      round(precio, 2) if precio else None,
                    "Trailing P/E": round(trailing, 2) if trailing else None,
                    "Forward P/E":  round(forward,  2) if forward  else None,
                })

            except Exception:
                pass

    df = pd.DataFrame(resultados)
    return df.reset_index(drop=True) if not df.empty else df


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class PERScanner:
    """
    Escanea el P/E Ratio de acciones de NYSE/NASDAQ o ByMA y genera rankings.
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_PER_TICKERS if mercado == "byma" else NYSE_PER_TICKERS

    def escanear(self, tipo_pe: str = "trailing", progress_callback=None) -> dict:
        """
        Descarga P/E Ratios y retorna rankings.

        Parameters
        ----------
        tipo_pe : str
            'trailing'  → usa Trailing P/E (últimos 12 meses)
            'forward'   → usa Forward P/E  (próximos 12 meses estimado)
            'ambos'     → muestra ambas columnas, ordena por Trailing P/E

        Returns
        -------
        dict {
            'bajo':   pd.DataFrame,   # P/E más bajos (potencial valor)
            'alto':   pd.DataFrame,   # P/E más altos (posible sobrevaluación)
            'todos':  pd.DataFrame,   # todos los resultados ordenados
            'col_pe': str,            # nombre de columna principal usada
        }
        """
        if progress_callback:
            progress_callback(0.05)

        df = _obtener_per_bulk(self.tickers)

        if progress_callback:
            progress_callback(0.92)

        cols_base = ["Ticker", "Nombre", "Sector", "Precio"]
        empty_cols = cols_base + ["Trailing P/E", "Forward P/E"]
        empty = pd.DataFrame(columns=empty_cols)

        if df.empty:
            return {"bajo": empty, "alto": empty, "todos": empty, "col_pe": "Trailing P/E"}

        # Determinar columna de ordenamiento
        if tipo_pe == "forward":
            col_pe = "Forward P/E"
        else:
            col_pe = "Trailing P/E"

        # Si tipo es 'ambos', mantenemos ambas columnas; si no, podemos igual mostrar ambas
        # Filtramos filas donde la columna principal tiene valor
        df_validos = df[df[col_pe].notna()].copy()

        if df_validos.empty:
            # Fallback: usar la otra columna
            col_pe     = "Forward P/E" if col_pe == "Trailing P/E" else "Trailing P/E"
            df_validos = df[df[col_pe].notna()].copy()

        bajo  = df_validos.sort_values(col_pe, ascending=True ).head(TOP_N).reset_index(drop=True)
        alto  = df_validos.sort_values(col_pe, ascending=False).head(TOP_N).reset_index(drop=True)
        todos = df_validos.sort_values(col_pe, ascending=True ).reset_index(drop=True)

        if progress_callback:
            progress_callback(1.0)

        return {"bajo": bajo, "alto": alto, "todos": todos, "col_pe": col_pe}
