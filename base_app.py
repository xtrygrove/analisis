import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema
import pandas as pd

# --- 1. Configuración de los Parámetros ---
# Tickers para el S&P 500, el rendimiento del bono a 10 años y VIX
tickers = ['^GSPC', '^TNX', '^VIX']
ticker_names = {'^GSPC': 'SPX', '^TNX': 'TNX', '^VIX': 'VIX'}

# Ventana para el cálculo móvil (rolling)
rolling_window_beta = 39
rolling_window_vol, long_window_vol = 13, 60 # For RV and Vol of Vol / For 3M RV
trading_days_per_year = 252 # For annualizing volatility

# --- 2. Descarga de Datos (para todos los tickers) ---
try:
  # Descargar los datos de precios de cierre ajustados para todos los tickers
  data = pd.read_csv(r'C:\Users\jfuentes\OneDrive - CINTAC S.A\Escritorio\EDO\Python\Proyectos\04_ANALISIS\analisis\datos_mercado.csv')

  if data.empty:
    # Lanzar un error si no se descargan datos
    raise ValueError("No se descargaron datos. Revisa los tickers o el rango de fechas.")

  # Rellenar valores faltantes (los fines de semana, por ejemplo)
  data.ffill(inplace=True)

  # Inicializar el indicador de descarga de datos
  data_downloaded = True

except Exception as e:
  print(f"Ocurrió un error al descargar los datos: {e}")
  # Establecer el indicador de descarga de datos a False si la descarga falla
  data_downloaded = False

# --- 3. Cálculo de Retornos y Cambios (para SPX, TNX y VIX) ---
# Verificar si los datos se descargaron correctamente antes de continuar
if data_downloaded and not data.empty:
  # Retornos porcentuales diarios para el S&P 500
  if 'SPX' in data.columns:
    data['spx_returns'] = data['SPX'].pct_change()
  else:
    print("Columna 'SPX' no encontrada. No se calcularán los retornos del SPX.")

  # Cambios absolutos diarios en el rendimiento del bono a 10 años
  if 'TNX' in data.columns:
    data['tnx_changes'] = data['TNX'].diff()
  else:
    print("Columna 'TNX' no encontrada. No se calcularán los cambios del TNX.")

  # Retornos porcentuales diarios para VIX
  if 'VIX' in data.columns:
    data['vix_returns'] = data['VIX'].pct_change()
  else:
    print("Columna 'VIX' no encontrada. No se calcularán los retornos del VIX.")

  # Eliminar filas con values NaN resultantes de los cálculos iniciales (pct_change, diff)
  cols_to_check_initial = [col for col in ['spx_returns', 'tnx_changes', 'vix_returns'] if col in data.columns]
  if cols_to_check_initial:
    data.dropna(subset=cols_to_check_initial, inplace=True)
  else:
    print("No hay columnas de retorno/cambio para eliminar NaNs.")

else:
  print("Saltando cálculos de retornos y cambios debido a un error de descarga de datos.")



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
    data['log_return'] = np.log(data['SPX'] / data['SPX'].shift(1))
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

# --- 8. Mostrar Gráficos en Orden ---
plt.style.use('dark_background') # Set style once

# Gráfico 1: Beta SPX vs TNX y Precio del SPX con Regímenes de Beta
if 'rolling_beta_spx_tnx' in data.columns and 'SPX' in data.columns:
  fig1, (ax1_beta_tnx, ax2_price_tnx) = plt.subplots(
      2, 1,
      figsize=(15, 10),
      sharex=True,
      gridspec_kw={'height_ratios': [1, 2]}
  )
  ax1_beta_tnx.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linewidth=1.5)
  ax1_beta_tnx.axhline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
  ax1_beta_tnx.set_title('Beta Móvil del S&P 500 vs Rendimiento del Bono a 10 Años', fontsize=16)
  ax1_beta_tnx.set_ylabel(f'Beta Móvil SPX/TNX {rolling_window_beta} días', fontsize=12)
  ax1_beta_tnx.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

  ax2_price_tnx.plot(data.index, data['SPX'], color='white', linewidth=1.5, label='Precio S&P 500')
  positive_beta_tnx_condition = data['rolling_beta_spx_tnx'] > 0
  negative_beta_tnx_condition = data['rolling_beta_spx_tnx'] <= 0
  min_y = data['SPX'].min()
  max_y = data['SPX'].max()
  ax2_price_tnx.fill_between(
      data.index, min_y*0.95, max_y*1.05,
      where=positive_beta_tnx_condition,
      color='red',
      alpha=0.25,
      label='Beta SPX/TNX > 0 (Correlación Positiva)'
  )
  ax2_price_tnx.fill_between(
      data.index, min_y*0.95, max_y*1.05,
      where=negative_beta_tnx_condition,
      color='green',
      alpha=0.25,
      label='Beta SPX/TNX < 0 (Correlación Negativa)'
  )
  ax2_price_tnx.set_ylabel('Precio del S&P 500 (SPX)', fontsize=12)
  ax2_price_tnx.set_xlabel('Fecha', fontsize=12)
  ax2_price_tnx.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  ax2_price_tnx.legend(loc='upper left')
  ax2_price_tnx.set_ylim(bottom=min_y*0.95, top=max_y*1.05)
  fig1.autofmt_xdate()
  ax2_price_tnx.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  ax2_price_tnx.xaxis.set_major_locator(mdates.AutoDateLocator())
  plt.tight_layout(pad=2.0)
  plt.show()
else:
  print("Saltando Gráfico 1: Faltan datos o columnas necesarias (rolling_beta_spx_tnx, SPX).")

# Gráfico 2: Beta SPX vs VIX
if 'rolling_beta_spx_vix' in data.columns:
  plt.figure(figsize=(15, 5))
  plt.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linewidth=1.5)
  plt.axhline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
  plt.title('Beta Móvil del S&P 500 vs VIX', fontsize=16)
  plt.ylabel(f'Beta Móvil SPX/VIX {rolling_window_beta} días', fontsize=12)
  plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  plt.show()
else:
  print("Saltando Gráfico 2: Faltan datos o columnas necesarias (rolling_beta_spx_vix, SPX).")

# Gráfico 3: Pendiente de las Betas Móviles
if 'beta_spx_tnx_slope' in data.columns or 'beta_spx_vix_slope' in data.columns:
  plt.figure(figsize=(15, 5))
  if 'beta_spx_tnx_slope' in data.columns:
    plt.plot(data.index, data['beta_spx_tnx_slope'], label='Pendiente Beta SPX/TNX', color='magenta', linewidth=1.5)
  if 'beta_spx_vix_slope' in data.columns:
    plt.plot(data.index, data['beta_spx_vix_slope'], label='Pendiente Beta SPX/VIX', color='cyan', linewidth=1.5)
  plt.title('Pendiente (Velocidad) de las Betas Móviles', fontsize=16)
  plt.xlabel('Fecha', fontsize=12)
  plt.ylabel('Cambio Diario en Beta', fontsize=12)
  plt.legend()
  plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  plt.gcf().autofmt_xdate()
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
  plt.tight_layout()
  plt.show()
else:
  print("Saltando Gráfico 3: No se encontraron columnas de pendiente para graficar.")

# Gráfico 4: Precio del SPX y las Betas Móviles SPX/TNX y SPX/VIX
if 'SPX' in data.columns and ('rolling_beta_spx_tnx' in data.columns or 'rolling_beta_spx_vix' in data.columns):
  fig4, ax_price_combo = plt.subplots(figsize=(15, 8))
  ax_betas_combo = ax_price_combo.twinx()

  ax_price_combo.plot(data.index, data['SPX'], color='white', linewidth=2.0, label='Precio S&P 500')
  ax_price_combo.set_ylabel('Precio del S&P 500 (SPX)', fontsize=12, color='white')
  ax_price_combo.tick_params(axis='y', labelcolor='white')
  ax_price_combo.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
  ax_price_combo.legend(loc='upper left')

  beta_lines = []
  if 'rolling_beta_spx_tnx' in data.columns:
    line1, = ax_betas_combo.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linestyle='--', linewidth=1.5, label='Beta SPX/TNX')
    beta_lines.append(line1)
  if 'rolling_beta_spx_vix' in data.columns:
    line2, = ax_betas_combo.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linestyle='--', linewidth=1.5, label='Beta SPX/VIX')
    beta_lines.append(line2)

  ax_betas_combo.set_ylabel('Beta Móvil', fontsize=12, color='white')
  ax_betas_combo.tick_params(axis='y', labelcolor='white')
  ax_betas_combo.axhline(0, color='grey', linestyle=':', linewidth=0.7, alpha=0.8)
  ax_betas_combo.legend(handles=beta_lines, loc='upper right')

  plt.title('Precio del S&P 500 y Betas Móviles SPX/TNX y SPX/VIX', fontsize=16)
  ax_price_combo.set_xlabel('Fecha', fontsize=12)
  fig4.autofmt_xdate()
  ax_price_combo.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  ax_price_combo.xaxis.set_major_locator(mdates.AutoDateLocator())

  plt.tight_layout()
  plt.show()
else:
  print("Saltando Gráfico 4: Faltan datos o columnas necesarias (SPX, rolling_beta_spx_tnx, rolling_beta_spx_vix).")

# Gráfico 5: Beta SPX/VIX Alternativa (Retorno SPX vs Cambio VIX)
if 'Rolling_Beta_Alt' in data.columns:
  plt.figure(figsize=(14, 7))
  plt.plot(data.index, data['Rolling_Beta_Alt'], color='purple', label='Rolling Beta')
  plt.axhline(0, color='yellow', linestyle='--', linewidth=1)
  plt.title('Rolling SPX/VIX Beta (SPX Return vs VIX Change)', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Beta (VIX points per 1% SPX move)', fontsize=12)
  plt.legend(loc='upper right')
  plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
  plt.tight_layout()
  plt.show()
else:
    print("Saltando Gráfico 5: Columna 'Rolling_Beta_Alt' no encontrada.")

# Gráfico 6: Volatilidad Realizada y Vol of Vol
if 'rv_20d' in data.columns and 'rv_3m' in data.columns and 'vol_of_vol' in data.columns:
  fig, ax1 = plt.subplots(figsize=(14, 7))
  ax1.plot(data.index, data['rv_20d'], color='purple', label=f'{rolling_window_vol}D rolling RV')
  ax1.plot(data.index, data['rv_3m'], color='gold', label=f'3M rolling RV')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Annualized Realized Volatility', color='lightgray')
  ax1.tick_params(axis='y', labelcolor='lightgray')

  ax2 = ax1.twinx()
  ax2.plot(data.index, data['vol_of_vol'], color='dimgray', label='Vol of vol')
  ax2.set_ylabel('Vol of Vol', color='dimgray')
  ax2.tick_params(axis='y', labelcolor='dimgray')

  plt.title(f'{ticker_names["^GSPC"]} - {rolling_window_vol}D rolling RV / {rolling_window_vol}D rolling Vol of Vol and 3M RV')
  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', facecolor='black', edgecolor='white')
  ax1.grid(False)
  ax2.grid(False)
  fig.tight_layout()
  plt.show()
else:
    print("Saltando Gráfico 6: Faltan columnas de volatilidad (rv_20d, rv_3m, vol_of_vol).")

# Gráfico 7: Distribución de Rendimientos Logarítmicos con Tendencia y Estadísticas
if 'log_return' in data.columns:
  log_returns = data['log_return'].dropna() # Ensure no NaNs for calculation and plotting

  if not log_returns.empty:
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
      fig.patch.set_facecolor('black')

      # Left Plot (Time Series)
      ax1.scatter(log_returns.index, log_returns, color='purple', label='Log Returns', zorder=5)
      x_numeric = np.arange(len(log_returns))
      coeffs = np.polyfit(x_numeric, log_returns, deg=3)
      p = np.poly1d(coeffs)
      ax1.plot(log_returns.index, p(x_numeric), color='yellow', linestyle='--', label='Trendline')
      ax1.set_title(f"[{ticker_names['^GSPC']}] Intraday Log Returns with Polynomial Trendline")
      ax1.set_xlabel('Date')
      ax1.set_ylabel('Log Returns')
      ax1.legend()
      ax1.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
      ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
      fig.autofmt_xdate(rotation=45)

      # Right Plot (Distribution)
      mean_val = log_returns.mean()
      median_val = log_returns.median()
      mode_val = log_returns.mode()[0] if not log_returns.mode().empty else float('nan') # Handle empty mode

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
      ax2.set_title(f"Distribution of [{ticker_names['^GSPC']}] Log Returns")
      ax2.set_xlabel('Log Returns')
      ax2.set_ylabel('Frequency')
      ax2.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')

      plt.tight_layout(pad=3.0)
      plt.show()
  else:
      print("Saltando Gráfico 7: No hay datos de log returns para graficar.")
else:
  print("Saltando Gráfico 7: Columna 'log_return' no encontrada.")


plt.style.use('dark_background')

if 'rolling_beta_spx_tnx' in data.columns and 'SPX' in data.columns:
    serie = data['rolling_beta_spx_tnx'].dropna()
    n = 3
    max_idx = argrelextrema(serie.values, np.greater, order=n)[0]
    min_idx = argrelextrema(serie.values, np.less, order=n)[0]
    puntos = np.sort(np.concatenate([max_idx, min_idx]))

    pendientes = []
    for i in range(len(puntos) - 1):
        i1, i2 = puntos[i], puntos[i+1]
        x1, x2 = serie.index[i1], serie.index[i2]
        y1, y2 = serie.iloc[i1], serie.iloc[i2]
        dias = (x2 - x1).days if hasattr(x2 - x1, 'days') else i2 - i1
        pendiente = (y2 - y1) / dias
        pendientes.append({
            'inicio': x1,
            'fin': x2,
            'beta_inicial': y1,
            'beta_final': y2,
            'dias': dias,
            'pendiente': pendiente
        })

    pendientes_df = pd.DataFrame(pendientes)
    st.subheader("Tabla de Pendientes por Impulso de la Beta SPX/TNX")
    st.dataframe(pendientes_df)

    st.subheader("Gráfico: Beta SPX/TNX y Precio SPX con Pendientes")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linewidth=1.5)
    ax1.axhline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
    ax1.set_title('Beta Móvil del S&P 500 vs Bono 10 Años', fontsize=16)
    ax1.set_ylabel('Beta SPX/TNX', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    for _, row in pendientes_df.iterrows():
        color_line = 'lime' if row['pendiente'] > 0 else 'orange'
        ax1.plot([row['inicio'], row['fin']], [row['beta_inicial'], row['beta_final']],
                 color=color_line, linewidth=2, linestyle='--', alpha=0.6)
        ax1.annotate(f"{row['pendiente']:.3f}", xy=(row['fin'], row['beta_final']),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8, color=color_line)

        ax2.plot([row['inicio'], row['fin']],
                 [data.loc[row['inicio'], 'SPX'], data.loc[row['fin'], 'SPX']],
                 color=color_line, linewidth=1.8, linestyle=':', alpha=0.5)
        ax2.annotate(f"{row['pendiente']:.3f}", xy=(row['fin'], data.loc[row['fin'], 'SPX']),
                     textcoords="offset points", xytext=(5, -10), ha='left', fontsize=8, color=color_line)

    ax2.plot(data.index, data['SPX'], color='white', linewidth=1.5, label='Precio S&P 500')
    min_y, max_y = data['SPX'].min(), data['SPX'].max()
    ax2.fill_between(data.index, min_y*0.95, max_y*1.05,
                     where=data['rolling_beta_spx_tnx'] > 0, color='red', alpha=0.25, label='Beta > 0')
    ax2.fill_between(data.index, min_y*0.95, max_y*1.05,
                     where=data['rolling_beta_spx_tnx'] <= 0, color='green', alpha=0.25, label='Beta < 0')
    ax2.set_ylabel('Precio SPX', fontsize=12)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='upper left')
    fig.autofmt_xdate()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    st.pyplot(fig)