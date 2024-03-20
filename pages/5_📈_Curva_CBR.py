# Filtrado de señales para archivos generados por la aplicación "AndroSensor", de Android
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

########## MODELO ##########
class Modelo:
    def __init__(self, df):
        self.df = df
        
    # Definir la función logística
    def logistic_function(self, x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Ajuste de la curva logística a los datos
    def fit_logistic_curve(self, x_data, y_data):
        popt, pcov = curve_fit(self.logistic_function, x_data, y_data, method='trf')
        return popt

    # Calcular la derivada de la función logística
    def logistic_derivative(self, x, L, k, x0):
        return (k * L * np.exp(-k * (x - x0))) / ((1 + np.exp(-k * (x - x0)))**2)

    # Encontrar el punto de inflexión
    def find_inflexion_point(self, popt):
        L, k, x0 = popt
        return x0

    # Calcular la ecuación de la recta tangente
    def tangent_line(self, popt, inflexion_point):
        L, k, x0 = popt
        slope = self.logistic_derivative(inflexion_point, L, k, x0)
        intercept = self.logistic_function(inflexion_point, L, k, x0) - slope * inflexion_point
        return slope, intercept
        
    def graficar_xy(self,x,y,x_tan,y_tan,titulo=None,yaxis_title=None,xaxis_title=None):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 line=dict(width=1)
                                )
                     )
        
        fig.add_trace(go.Scatter(x=x_tan,
                                 y=y_tan,
                                 line=dict(width=1)
                                )
                     )


        fig.update_layout(title=titulo,
                          yaxis_title=yaxis_title,
                          xaxis_title=xaxis_title,
                          height=600,
                          margin=dict(l=20, r=20, t=20, b=20)
                         )
        
        return fig

########## VISTA ##########
st.set_page_config(layout = 'wide',page_title='Curvas Penetración - Esfuerzo', page_icon="📈")
st.title('Curvas Penetración - Esfuerzo')

st.info('''App para el ajuste de puntos en curva de Penetración - Esfuerzo.
'''
       )

## Sidebar
with st.sidebar:
    #st.header('Filtrado de Señales')
    st.header('Subir archivo')
    file = st.file_uploader('Archivo en formato Excel',type='xlsx')
    if file is not None:
        # Leer el archivo subido a la página
        # Rutina para ignorar la primera fila si contiene la palabra 'sep'
        df = pd.read_excel(file)

        # Muestra una lista con las columnas como opciones
        columnas = df.columns
        columna_x = st.selectbox('Escoge los valores de Penetración',columnas)
        columna_y = st.selectbox('Escoge los valores de Esfuerzo',columnas)

        # Crea el objeto "modelo"
        modelo = Modelo(df)

        # Slider para seleccionar la frecuencia de corte
        x_data = df[columna_x].to_list()
        y_data = df[columna_y].to_list()
        
        # Ajustar curva logística a los datos
        popt = modelo.fit_logistic_curve(x_data, y_data)
        L, k, x0 = popt

        # Encontrar el punto de inflexión
        inflexion_point = modelo.find_inflexion_point(popt)

        # Calcular la ecuación de la recta tangente
        slope, intercept = modelo.tangent_line(popt, inflexion_point)

        x_tan = np.linspace(-intercept/slope, abs(x0)+abs(k)*.5,10)
        y_tan = slope*x_tan+intercept
        
        x_tan = x_tan.round(2)
        y_tan = y_tan.round(2)
        
        
        # Botón para descargar los datos filtrados
#         st.subheader('Descargar Datos Filtrados')
#         yFiltrado = modelo.filtrar_columna(columna,[fi,fc])
#         csv = modelo.convert_df(yFiltrado,columna)
#         st.download_button(
#             label="Descargar datos como CSV",
#             data=csv,
#             file_name='valores_filtrados.csv',
#             mime='text/csv')
        
        
        
if file is not None:

    # Para graficar los datos
    st.subheader('Gráfica')
    st.plotly_chart(modelo.graficar_xy(x_data,y_data,x_tan,y_tan, titulo="Penetración - Esfuerzo"),use_container_width=True)

