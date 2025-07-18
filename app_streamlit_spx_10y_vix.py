# -*- coding: utf-8 -*-
"""app_streamlit_spx/10y/vix.ipynb"""

#Cargar librerías
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns


st.set_page_config(layout="wide") # Configurar el layout para usar todo el ancho de la página

# Establecer el estilo de Matplotlib globalmente para todos los gráficos
plt.style.use('seaborn-v0_8-darkgrid')

st.title('Análisis para S&P 500')
st.title('Indicadores cuantitativos')

# --- Parámetros Fijos ---
# Tickers
tickers = ['^GSPC', '^TNX', '^VIX'] # '^GSPC', '^TNX', '^VIX'
ticker_names = {'^GSPC': 'SPX', '^TNX': 'TNX', '^VIX': 'VIX'}

# Ventana para el cálculo móvil (rolling)
rolling_window_beta = 13 # 4, 13, 33
rolling_window_vol, long_window_vol = 13, 60 # For RV and Vol of Vol / For 3M RV
trading_days_per_year = 252 # For annualizing volatility
period = 'ytd'    # 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 
interval = '1d'  # 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

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


# Gráfico 4: Precio del SPX y las Betas Móviles SPX/TNX y SPX/VIX
if 'SPX' in data.columns and ('rolling_beta_spx_tnx' in data.columns or 'rolling_beta_spx_vix' in data.columns):
    st.subheader("Gráfico 4: Precio SPX y Betas Móviles")
    fig4, ax_price = plt.subplots(figsize=(15, 8))
    ax_betas = ax_price.twinx()
    ax_price.plot(data.index, data['SPX'], color='white', linewidth=2.0, label='Precio S&P 500')
    ax_price.set_ylabel('Precio SPX', color='white')
    ax_price.tick_params(axis='y', labelcolor='white')
    beta_lines = []
    if 'rolling_beta_spx_tnx' in data.columns:
        line1, = ax_betas.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linestyle='--', linewidth=1.5, label='Beta SPX/TNX')
        beta_lines.append(line1)
    if 'rolling_beta_spx_vix' in data.columns:
        line2, = ax_betas.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linestyle='--', linewidth=1.5, label='Beta SPX/VIX')
        beta_lines.append(line2)
    ax_betas.set_ylabel('Beta Móvil', color='grey')
    ax_betas.tick_params(axis='y', labelcolor='grey')
    ax_betas.axhline(0, color='grey', linestyle=':', linewidth=0.7, alpha=0.8)
    ax_betas.legend(handles=beta_lines, loc='upper right')
    ax_price.set_xlabel('Fecha')
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig4.autofmt_xdate()
    st.pyplot(fig4)

# Gráfico 5: Beta SPX/VIX Alternativa
if 'Rolling_Beta_Alt' in data.columns:
    st.subheader("Gráfico 5: Beta SPX/VIX Alternativa")
    fig5, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Rolling_Beta_Alt'], color='purple', label='Rolling Beta')
    ax.axhline(0, color='yellow', linestyle='--', linewidth=1)
    ax.set_title('Rolling SPX/VIX Beta (SPX Return vs VIX Change)', fontsize=16)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Beta (VIX points per 1% SPX move)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray')
    st.pyplot(fig5)

# Gráfico 6: Volatilidad Realizada y Vol of Vol
if 'rv_20d' in data.columns and 'rv_3m' in data.columns and 'vol_of_vol' in data.columns:
    st.subheader("Gráfico 6: Volatilidad Realizada y Vol of Vol")
    fig6, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(data.index, data['rv_20d'], color='purple', label='20D rolling RV')
    ax1.plot(data.index, data['rv_3m'], color='gold', label='3M rolling RV')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Volatilidad Realizada Anualizada', color='lightgray')
    ax1.tick_params(axis='y', labelcolor='lightgray')
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['vol_of_vol'], color='dimgray', label='Vol of Vol')
    ax2.set_ylabel('Vol of Vol', color='dimgray')
    ax2.tick_params(axis='y', labelcolor='dimgray')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(False)
    ax2.grid(False)
    st.pyplot(fig6)

# Gráfico 7: Distribución de Rendimientos Logarítmicos
if 'log_return' in data.columns:
    log_returns = data['log_return'].dropna()
    if not log_returns.empty:
        st.subheader("Gráfico 7: Distribución de Rendimientos Logarítmicos")
        fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig7.patch.set_facecolor('black')
        ax1.scatter(log_returns.index, log_returns, color='purple', label='Log Returns', zorder=5)
        x_numeric = np.arange(len(log_returns))
        coeffs = np.polyfit(x_numeric, log_returns, deg=3)
        p = np.poly1d(coeffs)
        ax1.plot(log_returns.index, p(x_numeric), color='yellow', linestyle='--', label='Trendline')
        ax1.set_title('Log Returns with Polynomial Trendline')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Log Returns')
        ax1.legend()
        ax1.grid(True, linestyle='-', linewidth=0.5, color='gray')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig7.autofmt_xdate(rotation=45)
        mean_val = log_returns.mean()
        median_val = log_returns.median()
        mode_val = log_returns.mode()[0] if not log_returns.mode().empty else float('nan')
        sns.histplot(log_returns, bins=15, ax=ax2, color='purple', edgecolor='#300030',
                     kde=True, line_kws={'color': 'purple', 'linewidth': 2},
                     alpha=0.6, stat='frequency')
        ax2.axvline(mean_val, color='white', linestyle='-', linewidth=2)
        ax2.axvline(median_val, color='black', linestyle='--', linewidth=2)
        if not np.isnan(mode_val):
            ax2.axvline(mode_val, color='black', linestyle='-.', linewidth=2)
        legend_elements = [
            Line2D([0], [0], color='white', lw=2, linestyle='-', label=f'Mean:   {mean_val:.4f}'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', label=f'Median: {median_val:.4f}')
        ]
        if not np.isnan(mode_val):
            legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-.', label=f'Mode:    {mode_val:.4f}'))
        ax2.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white')
        ax2.set_title("Distribución de Log Returns")
        ax2.set_xlabel('Log Returns')
        ax2.set_ylabel('Frecuencia')
        ax2.grid(True, linestyle='-', linewidth=0.5, color='gray')
        st.pyplot(fig7)

st.write("---")
st.write("Esta información es solo para fines educativos y no debe considerarse asesoramiento financiero. El rendimiento pasado no garantiza resultados futuros.")
st.write("Desarrollado con Streamlit, yfinance, pandas y matplotlib.")
st.write("Eduardo Fuentes 2025")