# Filtrado de señales para archivos generados por la aplicación "AndroSensor", de Android
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft
from scipy import signal

########## MODELO ##########
class Modelo:
    def __init__(self, df):
        self.df = df
        
        self.t = pd.to_datetime(df['YYYY-MO-DD HH-MI-SS_SSS'],format='%Y-%m-%d %H:%M:%S:%f')
        df['DS'] = self.t
        self.ts = self.t.diff().dt.total_seconds().mean()
        self.fs = 1/self.ts
        
    def transformada_fourier(self, y, ts):
        n = len(y)
        yFourier = fft(y)
        xFourier = np.linspace(0.0, 1.0/(2.0*ts), n//2)
        yFourier = 2.0/n * np.abs(yFourier[0:n//2])
        return xFourier, yFourier

    def butter_bandpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = [x / nyq for x in cutoff]
        b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
        y = signal.filtfilt(b, a, data)
        return y
    
    def filtrar_columna(self,columna,fc):
        y=df[columna].values
        return self.butter_bandpass_filter(y,[0.1,fc],self.fs)
    
    def graficar_xy(self,x,y,titulo=None,yaxis_title=None,xaxis_title=None):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 line=dict(width=1)
                                )
                     )

        fig.update_layout(title=titulo,
                          yaxis_title=yaxis_title,
                          xaxis_title=xaxis_title,
                          height=200,
                          margin=dict(l=20, r=20, t=20, b=20)
                         )
        
        return fig
    
    @st.cache
    def convert_df(self,y,name):
        return pd.Series(y,name=name,index=self.t).to_csv().encode('utf-8')
    
    def graficar_datos_filtrados(self,columna,fc):
        y = self.filtrar_columna(columna,fc)
        x = df['DS']
        
        return self.graficar_xy(x,y,yaxis_title=columna)
    
    def graficar_columna(self,columna,titulo=None):
        x = df['DS']
        y = df[columna]
        yaxis_title=columna
        return self.graficar_xy(x,y,titulo,yaxis_title)
    
    def graficar_espectro(self,columna):
        y = df[columna].values
        y = y - y.mean()
        xFourier, yFourier = self.transformada_fourier(y, self.ts)
        
        return self.graficar_xy(xFourier,yFourier,yaxis_title='Amplitud',xaxis_title='Frecuencia (Hz)')

########## VISTA ##########
st.set_page_config(layout = 'wide')
st.title('Filtrado de Señales')

datosCol, espacioCol, graficosCol = st.columns((3,1,6))

with datosCol:
    st.info('App para el filtrado de señales, desde 0.1 Hz hasta la frecuencia deseada de corte')
    st.header('Subir archivo')
    file = st.file_uploader('Archivo en formato CSV',type='csv')
    if file is not None:
        # Leer el archivo subido a la página
        df = pd.read_csv(file,engine='python',sep=None)

        # Muestra una lista con las columnas como opciones
        columnas = df.columns
        columna = st.selectbox('Escoge una columna',columnas)

        # Crea el objeto "modelo"
        modelo = Modelo(df)

        # Slider para seleccionar la frecuencia de corte
        fs = modelo.fs
        fc = st.slider(label='Escoger la frecuencia de corte',
                             min_value = 0.1,
                             max_value = fs/2-0.1,
                             step = 0.1,
                             format = '%f',
                            )
        
        # Botón para descargar los datos filtrados
        st.subheader('Descargar Datos Filtrados')
        yFiltrado = modelo.filtrar_columna(columna,fc)
        csv = modelo.convert_df(yFiltrado,columna)
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='valores_filtrados.csv',
            mime='text/csv')
        
with graficosCol:
    if file is not None:
        # Para graficar los datos sin filtrar
        st.subheader('Datos sin Filtrar')
        st.plotly_chart(modelo.graficar_columna(columna),use_container_width=True)
        
        # Para graficar los datos filtrados
        st.subheader('Datos Filtrados')
        st.plotly_chart(modelo.graficar_datos_filtrados(columna,fc),use_container_width=True)

        # Para graficar el espectro de Fourier
        st.subheader('Espectro de Fourier')
        st.plotly_chart(modelo.graficar_espectro(columna),use_container_width=True)

