import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import io

MAX_SIZE_MB=5

st.set_page_config(page_title="Forecast Grupasa - Multi SKU", layout="wide")

st.title("📦 Forecast de Demanda - GRUPASA")
st.markdown("""
Sube un archivo CSV o Excel con tus datos históricos de ventas y genera pronósticos para todos los SKUs presentes.
""")

# ======================================
# 1️⃣ CARGA DE DATOS
# ======================================
uploaded_file = st.file_uploader(
    label=f"Sube tu archivo CSV o Excel (máx. {MAX_SIZE_MB} MB)",
    type=['csv', 'xlsx'],
    help=f"El archivo no debe superar {MAX_SIZE_MB} MB. Archivos más grandes serán rechazados."
)

st.markdown("""
📄 **Instrucciones para tu archivo de datos**

Para que la app genere el pronóstico correctamente, tu archivo CSV o Excel debe contener **estas columnas obligatorias**:

- `sku` → Identificador único del producto o SKU.
- `fecha` → Fecha de la venta o registro en formato `YYYY-MM-DD`.
- `demanda_real` → Cantidad vendida o demanda diaria.

Opcionalmente puedes incluir columnas adicionales como `categoria`, `sector` o `precio_unitario`, pero **estas tres son indispensables**.

⚠️ **Importante:** los nombres de las columnas deben coincidir exactamente (`sku`, `fecha`, `demanda_real`) para que la app funcione correctamente.
""")

if uploaded_file:
    # Validar tamaño
    file_size = uploaded_file.size / (1024 * 1024)  # Convertir bytes a MB
    if file_size > MAX_SIZE_MB:
        st.error(f"El archivo es demasiado grande ({file_size:.2f} MB). El límite es {MAX_SIZE_MB} MB.")
        uploaded_file = None
    else:
        # Leer archivo subido
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Archivo cargado correctamente!")
else:
    # Usar dataset de ejemplo (sintético)
    df = pd.read_csv("data/datos_grupasa.csv")
    st.info("Usando dataset sintético de ejemplo.")

# ======================================
# 2️⃣ VALIDACIÓN COLUMNAS
# ======================================
required_cols = ['sku', 'fecha', 'demanda_real']
for col in required_cols:
    if col not in df.columns:
        st.error(f"El dataset debe contener la columna '{col}'")
        st.stop()

# ======================================
# 3️⃣ PARÁMETROS DE FORECAST
# ======================================
st.sidebar.header("Parámetros del Forecast")
periods = st.sidebar.number_input("Horizonte a pronosticar (días)", min_value=7, max_value=365, value=30, step=7)
seasonality_mode = st.sidebar.selectbox("Modo de estacionalidad", ['additive', 'multiplicative'])

# ======================================
# 4️⃣ FORECAST POR SKU
# ======================================
skus = df['sku'].unique()
all_forecasts = []

st.subheader("📊 Pronósticos por SKU")
tabs = st.tabs([str(sku) for sku in skus])

for i, sku in enumerate(skus):
    with tabs[i]:
        st.markdown(f"### SKU: {sku}")

        df_sku = df[df['sku'] == sku].copy()
        df_sku['ds'] = pd.to_datetime(df_sku['fecha'])
        df_sku['y'] = df_sku['demanda_real']

        # Entrenar Prophet
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode=seasonality_mode)
        m.fit(df_sku[['ds','y']])

        # Predecir
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        forecast['sku'] = sku
        all_forecasts.append(forecast)

        # Gráfico interactivo
        st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

        # Componentes
        st.pyplot(m.plot_components(forecast))

        # Tabla de predicciones
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        st.dataframe(forecast_display)

# ======================================
# 5️⃣ DESCARGA DE TODOS LOS PRONÓSTICOS
# ======================================
st.subheader("📥 Descargar pronósticos de todos los SKUs")
df_all_forecast = pd.concat(all_forecasts)
excel_buffer = io.BytesIO()
df_all_forecast.to_excel(excel_buffer, index=False)
st.download_button(
    label="Descargar pronósticos (Excel)",
    data=excel_buffer,
    file_name="forecast_all_skus.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success("✅ Forecast generado para todos los SKUs.")
