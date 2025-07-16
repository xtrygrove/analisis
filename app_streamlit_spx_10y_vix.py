# -*- coding: utf-8 -*-
"""app_streamlit_spx/10y/vix.ipynb"""

import streamlit as st
import yfinance as yf
import pandas as pd
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
    # Calcular la covarianza móvil entre los retornos del SPX y los cambios en el rendimiento del bono
    rolling_cov_spx_tnx = data['spx_returns'].rolling(window=rolling_window_beta).cov(data['tnx_changes'])

    # Calcular la varianza móvil de los cambios en el rendimiento del bono
    rolling_var_tnx = data['tnx_changes'].rolling(window=rolling_window_beta).var()

    # Calcular la beta móvil SPX/TNX
    # Evitar división por cero si la varianza es 0 o NaN (cuando no hay suficientes datos)
    # Usamos .loc para evitar SettingWithCopyWarning si data es una vista
    data.loc[:, 'rolling_beta_spx_tnx'] = rolling_cov_spx_tnx / rolling_var_tnx
else:
    st.warning("Saltando cálculo de beta móvil SPX/TNX debido a la falta de datos o columnas necesarias.")

if 'spx_returns' in data.columns and 'vix_returns' in data.columns:
     # Calcular la covarianza móvil entre los retornos del SPX y los retornos del VIX
    rolling_cov_spx_vix = data['spx_returns'].rolling(window=rolling_window_beta).cov(data['vix_returns'])

    # Calcular la varianza móvil de los retornos del VIX
    rolling_var_vix = data['vix_returns'].rolling(window=rolling_window_beta).var()

    # Calcular la beta móvil SPX/VIX
    # Evitar división por cero si la varianza es 0 o NaN
    data.loc[:, 'rolling_beta_spx_vix'] = rolling_cov_spx_vix / rolling_var_vix
else:
    st.warning("Saltando cálculo de beta móvil SPX/VIX debido a la falta de datos o columnas necesarias.")

# Eliminar filas con valores NaN resultantes de los cálculos móviles (las primeras 'rolling_window_beta' - 1 filas)
# Solo hacemos dropna si las columnas existen
cols_to_check_beta = []
if 'rolling_beta_spx_tnx' in data.columns:
    cols_to_check_beta.append('rolling_beta_spx_tnx')
if 'rolling_beta_spx_vix' in data.columns:
    cols_to_check_beta.append('rolling_beta_spx_vix')

if cols_to_check_beta:
    data.dropna(subset=cols_to_check_beta, inplace=True)
else:
    st.info("Ninguna columna de beta calculada para eliminar NaNs.")

# --- 5. Cálculo de la Pendiente (Velocidad) de la Beta Móvil ---

# Verificar si las columnas de beta existen antes de calcular la pendiente
if 'rolling_beta_spx_tnx' in data.columns:
    # Calcular la pendiente (velocidad) de la beta móvil SPX/TNX
    data.loc[:, 'beta_spx_tnx_slope'] = data['rolling_beta_spx_tnx'].diff()
else:
    st.warning("Columna 'rolling_beta_spx_tnx' no encontrada. No se calculará la pendiente SPX/TNX.")

if 'rolling_beta_spx_vix' in data.columns:
    # Calcular la pendiente (velocidad) de la beta móvil SPX/VIX
    data.loc[:, 'beta_spx_vix_slope'] = data['rolling_beta_spx_vix'].diff()
else:
     st.warning("Columna 'rolling_beta_spx_vix' no encontrada. No se calculará la pendiente SPX/VIX.")

# Eliminar filas con valores NaN resultantes del diff() (el primer valor de cada pendiente)
# Solo hacemos dropna si las columnas de pendiente existen
cols_to_check_slope = []
if 'beta_spx_tnx_slope' in data.columns:
    cols_to_check_slope.append('beta_spx_tnx_slope')
if 'beta_spx_vix_slope' in data.columns:
    cols_to_check_slope.append('beta_spx_vix_slope')

if cols_to_check_slope:
    data.dropna(subset=cols_to_check_slope, inplace=True)
else:
    st.info("Ninguna columna de pendiente calculada para eliminar NaNs.")

# --- 6. Gráficos ---
# st.header('Visualizaciones') # Eliminado, cada gráfico tendrá su subheader

# Gráfico 1: Beta SPX vs TNX y Precio del SPX con Regímenes de Beta
if 'rolling_beta_spx_tnx' in data.columns and 'SPX' in data.columns:
    st.subheader('Beta SPX/TNX y Precio del SPX')

    fig1, (ax1_beta_tnx, ax2_price_tnx) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True, # Sincronizar el eje de fechas
        gridspec_kw={'height_ratios': [1, 2]} # Dar más espacio al gráfico de precios
    )

    # --- Gráfico Superior (Beta SPX/TNX) ---
    ax1_beta_tnx.plot(data.index, data['rolling_beta_spx_tnx'], color='blue', linewidth=1.5)
    ax1_beta_tnx.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.8)
    ax1_beta_tnx.set_title(f'Beta Móvil {rolling_window_beta}) del S&P 500 vs Rendimiento del Bono a 10 Años', fontsize=14)
    ax1_beta_tnx.set_ylabel('Beta Móvil SPX/TNX', fontsize=10)
    ax1_beta_tnx.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


    # --- Gráfico Inferior (Precio del SPX con Regímenes de Beta SPX/TNX) ---
    ax2_price_tnx.plot(data.index, data['SPX'], color='steelblue', linewidth=1.5, label='Precio S&P 500')

    # Definir las condiciones para colorear el fondo según la beta SPX/TNX
    positive_beta_tnx_condition = data['rolling_beta_spx_tnx'] > 0
    negative_beta_tnx_condition = data['rolling_beta_spx_tnx'] <= 0

    # Colorear el fondo según la beta SPX/TNX
    y_min_fill_tnx = data['SPX'].min() * 0.9
    y_max_fill_tnx = data['SPX'].max() * 1.1

    if positive_beta_tnx_condition.any():
         ax2_price_tnx.fill_between(
            data.index, y_min_fill_tnx, y_max_fill_tnx,
            where=positive_beta_tnx_condition,
            color='green',
            alpha=0.25,
            interpolate=True,
            label='Beta SPX/TNX > 0 (Correlación Positiva)'
        )

    if negative_beta_tnx_condition.any():
        ax2_price_tnx.fill_between(
            data.index, y_min_fill_tnx, y_max_fill_tnx,
            where=negative_beta_tnx_condition,
            color='red',
            alpha=0.25,
            interpolate=True,
            label='Beta SPX/TNX < 0 (Correlación Negativa)'
        )


    ax2_price_tnx.set_ylabel('Precio del S&P 500 (SPX)', fontsize=10)
    ax2_price_tnx.set_xlabel('Fecha', fontsize=10)
    ax2_price_tnx.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2_price_tnx.legend(loc='upper left')

    # --- Finalización y Muestra del Gráfico 1 ---
    fig1.autofmt_xdate()
    ax2_price_tnx.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2_price_tnx.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.tight_layout(pad=2.0) # Añadir padding entre los gráficos
    st.pyplot(fig1)
    plt.close(fig1) # Cerrar la figura para liberar memoria
    st.markdown("""
    **Interpretación del Gráfico 1:**
    *   El gráfico superior muestra la beta móvil del S&P 500 (SPX) con respecto al rendimiento del bono del Tesoro a 10 años (TNX). Una beta positiva (línea magenta sobre cero) sugiere que el SPX tiende a moverse en la misma dirección que los rendimientos de los bonos. Una beta negativa sugiere un movimiento en direcciones opuestas.
    *   El gráfico inferior muestra el precio del SPX. El fondo se colorea de verde cuando la beta SPX/TNX es positiva (SPX y TNX se mueven juntos) y de rojo cuando es negativa (SPX y TNX se mueven en oposición). Esto ayuda a visualizar cómo se comporta el mercado de acciones en diferentes regímenes de correlación con los tipos de interés.
    """)

else:
    st.warning("Saltando Gráfico 1: Faltan datos o columnas necesarias (rolling_beta_spx_tnx, SPX).")


# Gráfico 2: Beta SPX vs VIX y Precio del SPX con Regímenes de Beta
if 'rolling_beta_spx_vix' in data.columns and 'SPX' in data.columns:
    st.subheader('Beta SPX/VIX y Precio del SPX') # Estilo ya aplicado globalmente

    # Modificado para mostrar solo un gráfico: Beta SPX/VIX
    fig2, ax1_beta_vix = plt.subplots(figsize=(12, 5))

    # --- Gráfico Superior (Beta SPX/VIX) ---
    ax1_beta_vix.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linewidth=1.5)
    ax1_beta_vix.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.8)
    ax1_beta_vix.set_title(f'Beta Móvil {rolling_window_beta} del S&P 500 vs VIX', fontsize=14)
    ax1_beta_vix.set_ylabel('Beta Móvil SPX/VIX', fontsize=10)
    ax1_beta_vix.set_xlabel('Fecha', fontsize=10) # Añadir etiqueta X ya que es el único gráfico
    ax1_beta_vix.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # --- Finalización y Muestra del Gráfico 2 ---
    fig2.autofmt_xdate()
    ax1_beta_vix.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1_beta_vix.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.tight_layout(pad=2.0) # Añadir padding entre los gráficos
    st.pyplot(fig2)
    plt.close(fig2) # Cerrar la figura
    st.markdown("""
    **Interpretación del Gráfico 2:**
    *   Este gráfico muestra la beta móvil del S&P 500 (SPX) con respecto al VIX (índice de volatilidad). Históricamente, esta beta tiende a ser negativa (línea cian bajo cero), indicando que el SPX tiende a moverse en dirección opuesta al VIX (cuando el VIX sube, el SPX tiende a bajar, y viceversa).
    """)

else:
    st.warning("Saltando Gráfico 2: Faltan datos o columnas necesarias (rolling_beta_spx_vix, SPX).")


# Gráfico 3: Pendiente de las Betas Móviles
if 'beta_spx_tnx_slope' in data.columns or 'beta_spx_vix_slope' in data.columns:
    st.subheader('Pendiente (Velocidad) de las Betas Móviles') # Estilo ya aplicado globalmente
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    if 'beta_spx_tnx_slope' in data.columns:
        ax3.plot(data.index, data['beta_spx_tnx_slope'], label='Pendiente Beta SPX/TNX', color='magenta', linewidth=1.5)

    if 'beta_spx_vix_slope' in data.columns:
        ax3.plot(data.index, data['beta_spx_vix_slope'], label='Pendiente Beta SPX/VIX', color='cyan', linewidth=1.5)

    ax3.set_title('Pendiente (Velocidad) de las Betas Móviles', fontsize=14)
    ax3.set_xlabel('Fecha', fontsize=10)
    ax3.set_ylabel('Cambio Diario en Beta', fontsize=10)
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax3.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.8)


    # Formato de fechas para el eje X
    fig3.autofmt_xdate()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3) # Cerrar la figura
    st.markdown("""
    **Interpretación del Gráfico 3:**
    *   Este gráfico muestra la pendiente (o "velocidad") de las betas móviles calculadas anteriormente (Beta SPX/TNX y Beta SPX/VIX).
    *   Una pendiente positiva indica que la sensibilidad del SPX (su beta) a los cambios en TNX o VIX está aumentando.
    *   Una pendiente negativa indica que la sensibilidad está disminuyendo. Valores cercanos a cero sugieren que la beta es relativamente estable.
    """)
else:
    st.warning("Saltando Gráfico 3: No se encontraron columnas de pendiente para graficar.")


# --- Cálculo y Gráfico de la Correlación Móvil ---

# Verificar si las columnas necesarias existen para el cálculo de la correlación
if 'spx_returns' in data.columns and 'tnx_changes' in data.columns and 'vix_returns' in data.columns:
    # Calcular la correlación móvil entre los retornos de SPX/TNX y los retornos de SPX/VIX
    # Para calcular la correlación entre SPX/TNX y SPX/VIX, primero debemos calcular
    # los retornos de SPX/TNX y SPX/VIX. Dado que no tenemos esos ratios directos
    # y hemos calculado la beta que es una medida de sensibilidad, interpretaremos
    # la solicitud como la correlación entre la beta SPX/TNX y la beta SPX/VIX,
    # ya que la beta se deriva de las relaciones entre los retornos/cambios de estos activos.

    # Si lo que se busca es la correlación entre *retornos* del ratio SPX/TNX y SPX/VIX,
    # necesitaríamos calcular primero esos ratios, luego sus retornos, y finalmente la correlación móvil.
    # Basándonos en el código existente que calcula las Betas, parece más plausible
    # que la intención sea analizar la relación entre las Betas calculadas.

    # Vamos a calcular la correlación móvil entre las columnas de beta si existen.
    if 'rolling_beta_spx_tnx' in data.columns and 'rolling_beta_spx_vix' in data.columns: # rolling_window es 39
        st.subheader(f'Correlación Móvil (39 Días) entre Beta SPX/TNX y Beta SPX/VIX')

        # Calcular la correlación móvil de Pearson entre las dos columnas de beta
        rolling_correlation_betas = data['rolling_beta_spx_tnx'].rolling(window=rolling_window_beta).corr(data['rolling_beta_spx_vix'])

        # Añadir la nueva columna al DataFrame
        data.loc[:, 'rolling_corr_beta_tnx_vix'] = rolling_correlation_betas

        # Eliminar las filas NaN resultantes del cálculo de correlación
        data.dropna(subset=['rolling_corr_beta_tnx_vix'], inplace=True)


        # --- Gráfico de la Correlación Móvil ---
        fig_corr, ax_corr = plt.subplots(figsize=(12, 6)) # Estilo ya aplicado globalmente

        ax_corr.plot(data.index, data['rolling_corr_beta_tnx_vix'], color='gold', linewidth=1.5, label='Correlación Móvil (Beta SPX/TNX vs Beta SPX/VIX)')
        ax_corr.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.8) # Línea de correlación cero
        ax_corr.axhline(0.5, color='grey', linestyle=':', linewidth=0.7, alpha=0.6) # Línea de correlación moderada positiva
        ax_corr.axhline(-0.5, color='grey', linestyle=':', linewidth=0.7, alpha=0.6) # Línea de correlación moderada negativa


        ax_corr.set_title(f'Correlación Móvil {rolling_window_beta} entre Beta SPX/TNX y Beta SPX/VIX', fontsize=14)
        ax_corr.set_xlabel('Fecha', fontsize=10)
        ax_corr.set_ylabel('Coeficiente de Correlación', fontsize=10)
        ax_corr.legend()
        ax_corr.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Formato de fechas para el eje X
        fig_corr.autofmt_xdate()
        ax_corr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_corr.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr) # Cerrar la figura
        st.markdown("""
        **Interpretación del Gráfico de Correlación Móvil entre Betas:**
        *   Este gráfico muestra la correlación móvil entre la Beta SPX/TNX y la Beta SPX/VIX.
        *   Una correlación positiva (cercana a 1) indica que las dos betas tienden a moverse en la misma dirección; es decir, la sensibilidad del SPX al TNX y al VIX cambian de forma similar.
        *   Una correlación negativa (cercana a -1) indica que las betas tienden a moverse en direcciones opuestas. Una correlación cercana a 0 sugiere poca relación lineal entre los cambios de las dos betas.
        """)

    else:
        st.warning("Saltando cálculo y gráfico de correlación: Las columnas 'rolling_beta_spx_tnx' o 'rolling_beta_spx_vix' no existen.")
else:
    st.warning("Saltando cálculo y gráfico de correlación: Faltan columnas de retorno/cambio necesarias (spx_returns, tnx_changes, vix_returns).")


# Gráfico 4: Precio del SPX y las Betas Móviles SPX/TNX y SPX/VIX
if 'SPX' in data.columns and ('rolling_beta_spx_tnx' in data.columns or 'rolling_beta_spx_vix' in data.columns):
    st.subheader('Precio del SPX y Betas Móviles') # Estilo ya aplicado globalmente
    fig4, ax_price_combo = plt.subplots(figsize=(12, 8))
    ax_betas_combo = ax_price_combo.twinx() # Crear un eje y secundario para las betas

    # --- Gráfico del Precio SPX ---
    ax_price_combo.plot(data.index, data['SPX'], color='steelblue', linewidth=2.0, label='Precio S&P 500')
    ax_price_combo.set_ylabel('Precio del S&P 500 (SPX)', fontsize=10, color='steelblue')
    ax_price_combo.tick_params(axis='y', labelcolor='steelblue')
    ax_price_combo.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_price_combo.legend(loc='upper left')


    # --- Gráfico de las Betas en el Eje Secundario ---
    beta_lines = []
    if 'rolling_beta_spx_tnx' in data.columns:
        line1, = ax_betas_combo.plot(data.index, data['rolling_beta_spx_tnx'], color='magenta', linestyle='--', linewidth=1.5, label='Beta SPX/TNX (39 Días)')
        beta_lines.append(line1)
    if 'rolling_beta_spx_vix' in data.columns:
        line2, = ax_betas_combo.plot(data.index, data['rolling_beta_spx_vix'], color='cyan', linestyle='--', linewidth=1.5, label='Beta SPX/VIX (39 Días)')
        beta_lines.append(line2)

    ax_betas_combo.set_ylabel('Beta Móvil', fontsize=10, color='grey')
    ax_betas_combo.tick_params(axis='y', labelcolor='grey')
    ax_betas_combo.axhline(0, color='grey', linestyle=':', linewidth=0.7, alpha=0.8) # Línea de beta cero
    ax_betas_combo.legend(handles=beta_lines, loc='upper right')

    # --- Configuración General del Gráfico ---
    ax_betas_combo.set_title('Precio del S&P 500 y Betas Móviles SPX/TNX y SPX/VIX', fontsize=14)
    ax_price_combo.set_xlabel('Fecha', fontsize=10)
    fig4.autofmt_xdate() # Formato automático de fechas
    ax_price_combo.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_price_combo.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout() # Ajustar el diseño para evitar superposiciones
    st.pyplot(fig4)
    plt.close(fig4) # Cerrar la figura
    st.markdown("""
    **Interpretación del Gráfico 5 (Precio SPX y Betas Móviles):**
    *   Este gráfico combina la visualización del precio del S&P 500 (SPX, eje izquierdo, línea azul) con las dos betas móviles (Beta SPX/TNX en magenta y Beta SPX/VIX en cian, eje derecho).
    *   Permite observar directamente cómo evoluciona el precio del SPX mientras cambian sus sensibilidades (betas) al rendimiento de los bonos (TNX) y a la volatilidad (VIX).
    *   La línea de puntos horizontal en el eje de las betas marca el nivel cero, ayudando a identificar rápidamente si la sensibilidad es positiva o negativa.
    """)


else:
    st.warning("Saltando Gráfico 4: Faltan datos o columnas necesarias (SPX, rolling_beta_spx_tnx, rolling_beta_spx_vix).")

st.write("---")
st.write("Desarrollado con Streamlit, yfinance, pandas y matplotlib.")
