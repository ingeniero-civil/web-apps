# Filtrado de señales para archivos generados por la aplicación "AndroSensor", de Android
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from io import StringIO

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
    
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y

    def butter_bandpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = [x / nyq for x in cutoff]
        b, a = signal.butter(order, normal_cutoff, btype='bandpass', analog=False)
        y = signal.filtfilt(b, a, data)
        return y
    
    def filtrar_columna(self,columna,f):
        y=df[columna].values
        fi, fc = f
        if fi == 0:
            return self.butter_lowpass_filter(y,fc,self.fs)
        else:
            return self.butter_bandpass_filter(y,[fi,fc],self.fs)
    
    def graficar_xy(self,x,y,titulo=None,yaxis_title=None,xaxis_title=None,actividad=None):
        fig = go.Figure()
        
        if actividad is None:
            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     line=dict(width=1)
                                    )
                         )
        else:
            if actividad == 'TODAS':
                lista_actividades = df['ACTIVIDAD'].unique()
            else:
                lista_actividades = [actividad]
                
            for act in lista_actividades:
                mask = (df['ACTIVIDAD'] == act)
                fig.add_trace(go.Scatter(x=x.where(mask),
                                         y=y.where(mask),
                                         line=dict(width=1),
                                         name = act
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
    
    def graficar_datos_filtrados(self,columna,fc,actividad=None):
        y = pd.Series(self.filtrar_columna(columna,fc))
        x = df['DS']
        
        return self.graficar_xy(x,y,yaxis_title=columna,actividad=actividad)
    
    def graficar_columna(self,columna,titulo=None,actividad=None):
        x = df['DS']
        y = df[columna]
        yaxis_title=columna
        return self.graficar_xy(x,y,titulo,yaxis_title,actividad=actividad)
    
    @st.cache
    def get_espectro_df(self, columna, actividad = None):
        if actividad is None:
            y = df[columna].values
        else:
            if actividad == 'TODAS':
                lista_actividades = df['ACTIVIDAD'].unique()
            else:
                lista_actividades = [actividad]
            mask = df['ACTIVIDAD'].isin(lista_actividades)
            y = df[columna][mask].values
        y = y - y.mean()
        xFourier, yFourier = self.transformada_fourier(y, self.ts)
        return pd.DataFrame([xFourier,yFourier]).T.rename({0:'Frecuencia',1:'Fourier'},axis=1).to_csv(index=False).encode('utf-8')
    
    def graficar_espectro(self,columna, actividad=None):
        if actividad is None:
            y = df[columna].values
        else:
            if actividad == 'TODAS':
                lista_actividades = df['ACTIVIDAD'].unique()
            else:
                lista_actividades = [actividad]
            mask = df['ACTIVIDAD'].isin(lista_actividades)
            y = df[columna][mask].values
        #y = df[columna].values
        y = y - y.mean()
        xFourier, yFourier = self.transformada_fourier(y, self.ts)
        #xFourier = pd.Series(xFourier)
        #yFourier = pd.Series(yFourier)
        
        return self.graficar_xy(xFourier,yFourier,yaxis_title='Amplitud',xaxis_title='Frecuencia (Hz)')
    
    def graficar_espectro_filtrado(self,columna, fc, actividad=None):
        if actividad is None:
            y = df[columna].values
        else:
            if actividad == 'TODAS':
                lista_actividades = df['ACTIVIDAD'].unique()
            else:
                lista_actividades = [actividad]
            mask = df['ACTIVIDAD'].isin(lista_actividades)
            y = df[columna][mask].values
        #y = df[columna].values
        y = y - y.mean()
        if fc[0] == 0:
            y = self.butter_lowpass_filter(y,fc[1],self.fs)
        else:
            y = self.butter_bandpass_filter(y,fc,self.fs)
        xFourier, yFourier = self.transformada_fourier(y, self.ts)
        #xFourier = pd.Series(xFourier)
        #yFourier = pd.Series(yFourier)
        
        return self.graficar_xy(xFourier,yFourier,yaxis_title='Amplitud',xaxis_title='Frecuencia (Hz)')
    
    def graficar_espectrograma(self, columna):
        fig, ax = plt.subplots()
        fig.set_figheight(1)

        y = df[columna].values
        y = y - y.mean()
        f, t, Sxx = signal.spectrogram(y, self.fs)
        im = ax.pcolormesh(t, f, Sxx, shading='gouraud',cmap='viridis')
        #fig.colorbar(im,ax=ax)

        ds = df['DS']

        xt = ax.get_xticks()
        xtl = []
        for x in xt[:-1]:
            xtl.append(ds[int(x/self.ts)].strftime('%H:%M:%S'))

        ax.set_xticklabels(xtl,rotation='vertical')
        
        return fig

########## VISTA ##########
st.set_page_config(layout = 'wide',page_title='Filtrado de Señales', page_icon="📈")
st.title('Filtrado de Señales')

st.info('''App para el filtrado de señales, desde 0.1 Hz hasta la frecuencia deseada de corte.
'''
       )

## Sidebar
with st.sidebar:
    st.header('Filtrado de Señales')
    st.header('Subir archivo')
    file = st.file_uploader('Archivo en formato CSV',type='csv')
    if file is not None:
        # Leer el archivo subido a la página
        # Rutina para ignorar la primera fila si contiene la palabra 'sep'
        l = StringIO(file.getvalue().decode("utf-8")).readline()

        if 'sep=' in l:
            header=1
        else:
            header=0

        df = pd.read_csv(file,engine='python',sep=None,header=header)

        # Muestra una lista con las columnas como opciones
        columnas = df.columns
        columna = st.selectbox('Escoge una columna',columnas)

        # Crea el objeto "modelo"
        modelo = Modelo(df)

        # Slider para seleccionar la frecuencia de corte
        fs = modelo.fs
        st.write(round(fs/2,0)-0.1)
        fi, fc = st.slider(label='Escoger la frecuencia de corte',
                           max_value = round(fs/2,0)-0.1,
                           value=[0.0,round(fs/2,0)-0.1],
                           step = 0.1,
                           format = '%.1f',
                          )

        # Botón para descargar los datos filtrados
        st.subheader('Descargar Datos Filtrados')
        yFiltrado = modelo.filtrar_columna(columna,[fi,fc])
        csv = modelo.convert_df(yFiltrado,columna)
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='valores_filtrados.csv',
            mime='text/csv')
        
        
        
if file is not None:
    if 'ACTIVIDAD' in df.columns:
        lista = ['TODAS']
        lista.extend(list(df['ACTIVIDAD'].unique()))
        actividad = st.selectbox('Escoge una actividad',lista)

    else:
        actividad = None

    # Para graficar los datos sin filtrar
    st.subheader('Datos sin Filtrar')
    st.plotly_chart(modelo.graficar_columna(columna, actividad=actividad),use_container_width=True)

    # Para graficar los datos filtrados
    st.subheader('Datos Filtrados')
    st.plotly_chart(modelo.graficar_datos_filtrados(columna,[fi,fc], actividad=actividad),use_container_width=True)

    # Para graficar el espectro de Fourier
    st.subheader('Espectro de Fourier')
    st.plotly_chart(modelo.graficar_espectro(columna, actividad=actividad),use_container_width=True)

    # Para graficar el espectro de Fourier de los datos filtrados
    st.subheader('Espectro de Fourier de Señal Filtrada')
    st.plotly_chart(modelo.graficar_espectro_filtrado(columna, [fi,fc], actividad=actividad),use_container_width=True)

    # Botón para descargar el espectro de Fourier
    st.subheader('Descargar Espectro de Fourier')
    csv_espectro = modelo.get_espectro_df(columna, actividad = actividad)
    st.download_button(
        label="Descargar espectro como CSV",
        data=csv_espectro,
        file_name='espectro_fourier.csv',
        mime='text/csv')

    # Para graficar el espectrograma
    st.subheader('Espectrograma')
    st.pyplot(modelo.graficar_espectrograma(columna))
