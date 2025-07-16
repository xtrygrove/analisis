import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import seaborn as sns
from matplotlib.lines import Line2D
import warnings

# --- 1. Parámetros ---
tickers = ['^GSPC', '^TNX', '^VIX']
ticker_names = {'^GSPC': 'SPX', '^TNX': 'TNX', '^VIX': 'VIX'}
rolling_window_beta, smoothing = 39, 39
rolling_window_vol, long_window_vol = 13, 60
trading_days_per_year = 252

# --- 2. Descarga ---
# try:
#     data = yf.download(tickers, period='1y', interval='1d')['Close'].rename(columns=ticker_names)
#     if data.empty:
#         raise ValueError("No data downloaded.")
#     data.ffill(inplace=True)
# except Exception as e:
#     raise SystemExit(f"Error al descargar: {e}")

data = pd.read_csv('datos_mercado.csv') #archivos en csv para trabajar 

# --- 3. Retornos y Cambios ---
data['spx_returns'] = data['SPX'].pct_change() if 'SPX' in data else np.nan
data['tnx_changes'] = data['TNX'].diff() if 'TNX' in data else np.nan
data['vix_returns'] = data['VIX'].pct_change() if 'VIX' in data else np.nan

# Drop NaNs iniciales
data.dropna(subset=['spx_returns', 'tnx_changes', 'vix_returns'], how='all', inplace=True)

# --- Función utilitaria ---
def calc_rolling_beta(df, x, y, window, ema_span=None):
    if {x, y}.issubset(df.columns):
        cov = df[x].rolling(window).cov(df[y])
        var = df[y].rolling(window).var()
        beta = cov / var
        if ema_span:
            return beta.ewm(span=ema_span).mean()
        return beta
    else:
        warnings.warn(f"Skipping beta: columns {x} or {y} missing")
        return None

# --- 4. Betas ---
data['beta_spx_tnx'] = calc_rolling_beta(data, 'spx_returns', 'tnx_changes', rolling_window_beta)
data['beta_spx_tnx_ema'] = data['beta_spx_tnx'].ewm(span=smoothing).mean()
data['beta_spx_vix'] = calc_rolling_beta(data, 'spx_returns', 'vix_returns', rolling_window_beta)

# --- 5. Pendientes ---
data['beta_spx_tnx_slope'] = data['beta_spx_tnx'].diff()
data['beta_spx_vix_slope'] = data['beta_spx_vix'].diff()

# --- 6. Volatilidad ---
if 'SPX' in data:
    data['log_return'] = np.log(data['SPX'] / data['SPX'].shift(1))
    data['rv_20d'] = data['log_return'].rolling(rolling_window_vol).std() * np.sqrt(trading_days_per_year)
    data['rv_3m'] = data['log_return'].rolling(long_window_vol).std() * np.sqrt(trading_days_per_year)
    data['vol_of_vol'] = data['rv_20d'].rolling(rolling_window_vol).std()

# --- 7. Beta Alternativa ---
if {'SPX', 'VIX'}.issubset(data.columns):
    data['SPX_pct'] = data['SPX'].pct_change() * 100
    data['VIX_diff'] = data['VIX'].diff()
    cov_alt = data['SPX_pct'].rolling(rolling_window_vol).cov(data['VIX_diff'])
    var_alt = data['SPX_pct'].rolling(rolling_window_vol).var()
    data['beta_alt'] = cov_alt / var_alt

# --- Limpieza final ---
data.dropna(inplace=True)

# --- 8. Gráficos ---
with plt.style.context('dark_background'):
    if {'beta_spx_tnx', 'SPX'}.issubset(data.columns):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.plot(data.index, data['beta_spx_tnx'], color='magenta', lw=1.5)
        ax1.axhline(0, color='white', ls='--', lw=0.8)
        ax1.set_title('Beta SPX/TNX')
        ax2.plot(data.index, data['SPX'], color='white')
        ax2.fill_between(data.index, data['SPX'].min(), data['SPX'].max(),
                         where=data['beta_spx_tnx']>0, color='red', alpha=0.2, label='Beta>0')
        ax2.fill_between(data.index, data['SPX'].min(), data['SPX'].max(),
                         where=data['beta_spx_tnx']<=0, color='green', alpha=0.2, label='Beta<=0')
        ax2.legend()
        fig.autofmt_xdate()
        plt.show()

    if 'beta_spx_vix' in data:
        plt.figure(figsize=(14, 5))
        plt.plot(data.index, data['beta_spx_vix'], color='cyan')
        plt.title('Beta SPX/VIX')
        plt.show()

    if 'beta_spx_tnx_slope' in data or 'beta_spx_vix_slope' in data:
        plt.figure(figsize=(14, 5))
        if 'beta_spx_tnx_slope' in data:
            plt.plot(data.index, data['beta_spx_tnx_slope'], label='Slope SPX/TNX', color='magenta')
        if 'beta_spx_vix_slope' in data:
            plt.plot(data.index, data['beta_spx_vix_slope'], label='Slope SPX/VIX', color='cyan')
        plt.legend()
        plt.title('Slope of Betas')
        plt.show()

    if 'rv_20d' in data and 'rv_3m' in data and 'vol_of_vol' in data:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(data.index, data['rv_20d'], color='purple')
        ax1.plot(data.index, data['rv_3m'], color='gold')
        ax2 = ax1.twinx()
        ax2.plot(data.index, data['vol_of_vol'], color='gray')
        plt.title('Realized Volatility & Vol of Vol')
        plt.show()

    if 'beta_alt' in data:
        plt.figure(figsize=(14, 5))
        plt.plot(data.index, data['beta_alt'], color='purple')
        plt.title('Beta SPX/VIX Alternative')
        plt.show()

    if 'log_return' in data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.scatter(data.index, data['log_return'], color='purple')
        coeffs = np.polyfit(range(len(data['log_return'])), data['log_return'], 3)
        ax1.plot(data.index, np.polyval(coeffs, range(len(data['log_return']))), color='yellow')
        sns.histplot(data['log_return'], ax=ax2, bins=20, kde=True, color='purple')
        ax2.axvline(data['log_return'].mean(), color='white')
        plt.show()
