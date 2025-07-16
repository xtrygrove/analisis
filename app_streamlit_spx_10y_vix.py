# -*- coding: utf-8 -*-
"""app_streamlit_spx/10y/vix.ipynb"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide") # Configurar el layout para usar todo el ancho de la página

# Establecer el estilo de Matplotlib globalmente para todos los gráficos
plt.style.use('seaborn-v0_8-darkgrid')

st.title('Análisis de Betas Móviles del S&P 500')

# --- Parámetros Fijos ---
# Tickers
tickers = ['^GSPC', '^TNX', '^VIX']
ticker_names = {'^GSPC': 'SPX', '^TNX': 'TNX', '^VIX': 'VIX'}

# Ventana para el cálculo móvil (rolling)
rolling_window_beta = 39
rolling_window_vol, long_window_vol = 13, 60 # For RV and Vol of Vol / For 3M RV
trading_days_per_year = 252 # For annualizing volatility
period = '1y'
interval = '1d'

data = pd.DataFrame()

# --- 2. Descarga de Datos (para todos los tickers) ---
# st.header('Descarga de Datos') # Eliminado para simplificar la vista

@st.cache_data # Cachear la descarga de datos para evitar descargas repetidas
def download_data(tickers, period, interval):
    try:
        # Descargar los datos de precios de cierre ajustados para todos los tickers
        data = yf.download(tickers, period=period, interval=interval)['Close']
        data.rename(columns=ticker_names, inplace=True)

        if data.empty:
            return None, "No se descargaron datos. Revisa los tickers o el rango de fechas."

        # Rellenar valores faltantes (los fines de semana, por ejemplo)
        data.ffill(inplace=True)

        return data, None # Retorna el DataFrame y None para el error

    except Exception as e:
        return None, f"Ocurrió un error al descargar los datos: {e}" # Retorna None y el mensaje de error

# Descargar datos (se ejecuta siempre al cargar/refrescar la app)
data, download_error = download_data(tickers, period, interval)
if download_error:
    st.error(download_error)
    st.stop() # Detiene la ejecución si hay un error de descarga

# Verificar si los datos se descargaron correctamente antes de continuar
if not data.empty:
    # Retornos porcentuales diarios para el S&P 500
    if 'SPX' in data.columns:
        data['spx_returns'] = data['SPX'].pct_change()
    else:
        st.warning("Columna 'SPX' no encontrada. No se calcularán los retornos del SPX.")

    # Cambios absolutos diarios en el rendimiento del bono a 10 años
    if 'TNX' in data.columns:
        data['tnx_changes'] = data['TNX'].diff()
    else:
        st.warning("Columna 'TNX' no encontrada. No se calcularán los cambios del TNX.")

    # Retornos porcentuales diarios para VIX
    if 'VIX' in data.columns:
        data['vix_returns'] = data['VIX'].pct_change()
    else:
        st.warning("Columna 'VIX' no encontrada. No se calcularán los retornos del VIX.")


    # Eliminar filas con valores NaN resultantes de los cálculos iniciales (pct_change, diff)
    # Es importante que al menos una de las columnas 'spx_returns', 'tnx_changes', 'vix_returns' exista para evitar errores si alguna no se calculó.
    cols_to_check_initial = [col for col in ['spx_returns', 'tnx_changes', 'vix_returns'] if col in data.columns]
    if cols_to_check_initial:
        data.dropna(subset=cols_to_check_initial, inplace=True)
    else:
        st.info("No hay columnas de retorno/cambio para eliminar NaNs.")

else:
    st.warning("Saltando cálculos de retornos y cambios debido a un error de descarga de datos.")

# --- 4. Cálculo de la Beta Móvil (SPX/TNX y SPX/VIX) ---
# Verificar si los datos están listos para los cálculos de beta
if 'spx_returns' in data.columns and 'tnx_changes' in data.columns:
  rolling_cov_spx_tnx = data['spx_returns'].rolling(window=rolling_window_beta).cov(data['tnx_changes'])
  rolling_var_tnx = data['tnx_changes'].rolling(window=rolling_window_beta).var()
  data.loc[:, 'rolling_beta_spx_tnx'] = rolling_cov_spx_tnx / rolling_var_tnx
else:
  print("Saltando cálculo de beta móvil SPX/TNX debido a la falta de datos o columnas necesarias.")

if 'spx_returns' in data.columns and 'vix_returns' in data.columns:
  rolling_cov_spx_vix = data['spx_returns'].rolling(window=rolling_window_beta).cov(data['vix_returns'])
  rolling_var_vix = data['vix_returns'].rolling(window=rolling_window_beta).var()
  data.loc[:, 'rolling_beta_spx_vix'] = rolling_cov_spx_vix / rolling_var_vix
else:
  print("Saltando cálculo de beta móvil SPX/VIX debido a la falta de datos o columnas necesarias.")

# Eliminar filas con values NaN resultantes de los cálculos móviles
cols_to_check_beta = []
if 'rolling_beta_spx_tnx' in data.columns:
  cols_to_check_beta.append('rolling_beta_spx_tnx')
if 'rolling_beta_spx_vix' in data.columns:
  cols_to_check_beta.append('rolling_beta_spx_vix')

if cols_to_check_beta:
  data.dropna(subset=cols_to_check_beta, inplace=True)
else:
  print("Ninguna columna de beta calculada para eliminar NaNs.")

# --- 5. Cálculo de la Pendiente (Velocidad) de la Beta Móvil ---
if 'rolling_beta_spx_tnx' in data.columns:
  data.loc[:, 'beta_spx_tnx_slope'] = data['rolling_beta_spx_tnx'].diff()
else:
  print("Columna 'rolling_beta_spx_tnx' no encontrada. No se calculará la pendiente SPX/TNX.")

if 'rolling_beta_spx_vix' in data.columns:
  data.loc[:, 'beta_spx_vix_slope'] = data['rolling_beta_spx_vix'].diff()
else:
  print("Columna 'rolling_beta_spx_vix' no encontrada. No se calculará la pendiente SPX/VIX.")

cols_to_check_slope = []
if 'beta_spx_tnx_slope' in data.columns:
  cols_to_check_slope.append('beta_spx_tnx_slope')
if 'beta_spx_vix_slope' in data.columns:
  cols_to_check_slope.append('beta_spx_vix_slope')

if cols_to_check_slope:
  data.dropna(subset=cols_to_check_slope, inplace=True)
else:
  print("Ninguna columna de pendiente calculada para eliminar NaNs.")

# --- 6. Cálculo de Volatilidad Realizada y Vol of Vol ---
if 'SPX' in data.columns:
    data['log_return'] = np.log(data['SPX'] / data['SPX'].shift())
    data['rv_20d'] = data['log_return'].rolling(window=rolling_window_vol).std() * np.sqrt(trading_days_per_year)
    data['rv_3m'] = data['log_return'].rolling(window=long_window_vol).std() * np.sqrt(trading_days_per_year)
    data['vol_of_vol'] = data['rv_20d'].rolling(window=rolling_window_vol).std()
    data.dropna(subset=['rv_20d', 'rv_3m', 'vol_of_vol', 'log_return'], inplace=True) # Drop NaNs after these calculations
else:
    print("Columna 'SPX' no encontrada. No se calcularán métricas de volatilidad.")

# --- 7. Cálculo de Beta SPX/VIX Alternativa (usando Retorno SPX vs Cambio VIX) ---
if 'SPX' in data.columns and 'VIX' in data.columns:
    data['SPX_Return_percent'] = data['SPX'].pct_change() * 100
    data['VIX_Change_points'] = data['VIX'].diff()
    rolling_cov_alt = data['SPX_Return_percent'].rolling(window=rolling_window_vol).cov(data['VIX_Change_points'])
    # Using variance of SPX return here, as in the original alternative beta code
    rolling_var_alt = data['SPX_Return_percent'].rolling(window=rolling_window_vol).var()
    data['Rolling_Beta_Alt'] = rolling_cov_alt / rolling_var_alt
    data.dropna(subset=['SPX_Return_percent', 'VIX_Change_points', 'Rolling_Beta_Alt'], inplace=True) # Drop NaNs
else:
     print("Columnas 'SPX' o 'VIX' no encontradas. No se calculará la Beta SPX/VIX alternativa.")

# --- 6. Gráficos ---
plt.style.use('dark_background')

# Gráfico 1: Beta SPX vs TNX y Precio del SPX con Regímenes de Beta
if 'rolling_beta_spx_tnx' in data.columns and 'SPX' in data.columns:
    st.subheader("Gráfico 1: Beta SPX vs TNX y Precio del SPX")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linewidth=1.5)
    ax1.axhline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
    ax1.set_title('Beta Móvil del S&P 500 vs Rendimiento del Bono a 10 Años', fontsize=16)
    ax1.set_ylabel('Beta Móvil SPX/TNX', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    ax2.plot(data.index, data['SPX'], color='white', linewidth=1.5, label='Precio S&P 500')
    pos_cond = data['rolling_beta_spx_tnx'] > 0
    neg_cond = data['rolling_beta_spx_tnx'] <= 0
    min_y, max_y = data['SPX'].min(), data['SPX'].max()
    ax2.fill_between(data.index, min_y*0.95, max_y*1.05, where=pos_cond, color='red', alpha=0.25, label='Beta > 0')
    ax2.fill_between(data.index, min_y*0.95, max_y*1.05, where=neg_cond, color='green', alpha=0.25, label='Beta < 0')
    ax2.set_ylabel('Precio del S&P 500 (SPX)', fontsize=12)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='upper left')
    fig1.autofmt_xdate()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    st.pyplot(fig1)

# Gráfico 2: Beta SPX vs VIX
if 'rolling_beta_spx_vix' in data.columns:
    st.subheader("Gráfico 2: Beta SPX vs VIX")
    fig2, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linewidth=1.5)
    ax.axhline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_title('Beta Móvil del S&P 500 vs VIX', fontsize=16)
    ax.set_ylabel('Beta Móvil SPX/VIX', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig2)

# Gráfico 3: Pendiente de las Betas Móviles
if 'beta_spx_tnx_slope' in data.columns or 'beta_spx_vix_slope' in data.columns:
    st.subheader("Gráfico 3: Pendiente de las Betas Móviles")
    fig3, ax = plt.subplots(figsize=(15, 5))
    if 'beta_spx_tnx_slope' in data.columns:
        ax.plot(data.index, data['beta_spx_tnx_slope'], label='Pendiente Beta SPX/TNX', color='magenta')
    if 'beta_spx_vix_slope' in data.columns:
        ax.plot(data.index, data['beta_spx_vix_slope'], label='Pendiente Beta SPX/VIX', color='cyan')
    ax.set_title('Pendiente de las Betas Móviles', fontsize=16)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Cambio Diario en Beta')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig3.autofmt_xdate()
    st.pyplot(fig3)

st.write("---")
st.write("Desarrollado con Streamlit, yfinance, pandas y matplotlib.")
