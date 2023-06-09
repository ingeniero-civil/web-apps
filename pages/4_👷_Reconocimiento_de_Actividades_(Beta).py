# Filtrado de señales para archivos generados por la aplicación "AndroSensor", de Android
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from io import StringIO
import joblib, pickle
from scipy.signal import find_peaks
from scipy import stats
import os

########## MODELO ##########
class Modelo:
    def __init__(self, df):
        self.df = df
        
        try:
            self.t = pd.to_datetime(df['YYYY-MO-DD HH-MI-SS_SSS'],format='%Y-%m-%d %H:%M:%S:%f')
        except:
            self.t = pd.to_datetime(df['time'].astype('int64'),unit='ns')
        df['DS'] = self.t
        self.ts = self.t.diff().dt.total_seconds().median()
        self.fs = 1/self.ts
        
    def get_x_params(self,df1):
        x_list = []
        y_list = []
        z_list = []

        window_size = 250
        step_size = 125

        # creating overlaping windows of size window-size 100
        for i in range(0, df1.shape[0] - window_size, step_size):
            xs = df1['x'].values[i: i + window_size]
            ys = df1['y'].values[i: i + window_size]
            zs = df1['z'].values[i: i + window_size]

            x_list.append(xs)
            y_list.append(ys)
            z_list.append(zs)

        # Statistical Features on raw x, y and z in time domain
        X_test = pd.DataFrame()

        # mean
        X_test['x_mean'] = pd.Series(x_list).apply(lambda x: x.mean())
        X_test['y_mean'] = pd.Series(y_list).apply(lambda x: x.mean())
        X_test['z_mean'] = pd.Series(z_list).apply(lambda x: x.mean())

        # std dev
        X_test['x_std'] = pd.Series(x_list).apply(lambda x: x.std())
        X_test['y_std'] = pd.Series(y_list).apply(lambda x: x.std())
        X_test['z_std'] = pd.Series(z_list).apply(lambda x: x.std())

        # avg absolute diff
        X_test['x_aad'] = pd.Series(x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_test['y_aad'] = pd.Series(y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_test['z_aad'] = pd.Series(z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

        # min
        X_test['x_min'] = pd.Series(x_list).apply(lambda x: x.min())
        X_test['y_min'] = pd.Series(y_list).apply(lambda x: x.min())
        X_test['z_min'] = pd.Series(z_list).apply(lambda x: x.min())

        # max
        X_test['x_max'] = pd.Series(x_list).apply(lambda x: x.max())
        X_test['y_max'] = pd.Series(y_list).apply(lambda x: x.max())
        X_test['z_max'] = pd.Series(z_list).apply(lambda x: x.max())

        # max-min diff
        X_test['x_maxmin_diff'] = X_test['x_max'] - X_test['x_min']
        X_test['y_maxmin_diff'] = X_test['y_max'] - X_test['y_min']
        X_test['z_maxmin_diff'] = X_test['z_max'] - X_test['z_min']

        # median
        X_test['x_median'] = pd.Series(x_list).apply(lambda x: np.median(x))
        X_test['y_median'] = pd.Series(y_list).apply(lambda x: np.median(x))
        X_test['z_median'] = pd.Series(z_list).apply(lambda x: np.median(x))

        # median abs dev 
        X_test['x_mad'] = pd.Series(x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_test['y_mad'] = pd.Series(y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_test['z_mad'] = pd.Series(z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

        # interquartile range
        X_test['x_IQR'] = pd.Series(x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_test['y_IQR'] = pd.Series(y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_test['z_IQR'] = pd.Series(z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # negtive count
        X_test['x_neg_count'] = pd.Series(x_list).apply(lambda x: np.sum(x < 0))
        X_test['y_neg_count'] = pd.Series(y_list).apply(lambda x: np.sum(x < 0))
        X_test['z_neg_count'] = pd.Series(z_list).apply(lambda x: np.sum(x < 0))

        # positive count
        X_test['x_pos_count'] = pd.Series(x_list).apply(lambda x: np.sum(x > 0))
        X_test['y_pos_count'] = pd.Series(y_list).apply(lambda x: np.sum(x > 0))
        X_test['z_pos_count'] = pd.Series(z_list).apply(lambda x: np.sum(x > 0))

        # values above mean
        X_test['x_above_mean'] = pd.Series(x_list).apply(lambda x: np.sum(x > x.mean()))
        X_test['y_above_mean'] = pd.Series(y_list).apply(lambda x: np.sum(x > x.mean()))
        X_test['z_above_mean'] = pd.Series(z_list).apply(lambda x: np.sum(x > x.mean()))

        # number of peaks
        X_test['x_peak_count'] = pd.Series(x_list).apply(lambda x: len(find_peaks(x)[0]))
        X_test['y_peak_count'] = pd.Series(y_list).apply(lambda x: len(find_peaks(x)[0]))
        X_test['z_peak_count'] = pd.Series(z_list).apply(lambda x: len(find_peaks(x)[0]))

        # skewness
        X_test['x_skewness'] = pd.Series(x_list).apply(lambda x: stats.skew(x))
        X_test['y_skewness'] = pd.Series(y_list).apply(lambda x: stats.skew(x))
        X_test['z_skewness'] = pd.Series(z_list).apply(lambda x: stats.skew(x))

        # kurtosis
        X_test['x_kurtosis'] = pd.Series(x_list).apply(lambda x: stats.kurtosis(x))
        X_test['y_kurtosis'] = pd.Series(y_list).apply(lambda x: stats.kurtosis(x))
        X_test['z_kurtosis'] = pd.Series(z_list).apply(lambda x: stats.kurtosis(x))

        # energy
        X_test['x_energy'] = pd.Series(x_list).apply(lambda x: np.sum(x**2)/100)
        X_test['y_energy'] = pd.Series(y_list).apply(lambda x: np.sum(x**2)/100)
        X_test['z_energy'] = pd.Series(z_list).apply(lambda x: np.sum(x**2/100))

        # avg resultant
        X_test['avg_result_accl'] = [i.mean() for i in ((pd.Series(x_list)**2 + pd.Series(y_list)**2 + pd.Series(z_list)**2)**0.5)]

        # signal magnitude area
        X_test['sma'] =    pd.Series(x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list).apply(lambda x: np.sum(abs(x)/100)) \
                          + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/100))

        # converting the signals from time domain to frequency domain using FFT
        x_list_fft = pd.Series(x_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
        y_list_fft = pd.Series(y_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
        z_list_fft = pd.Series(z_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])

        # Statistical Features on raw x, y and z in frequency domain
        # FFT mean
        X_test['x_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: x.mean())
        X_test['y_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: x.mean())
        X_test['z_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: x.mean())

        # FFT std dev
        X_test['x_std_fft'] = pd.Series(x_list_fft).apply(lambda x: x.std())
        X_test['y_std_fft'] = pd.Series(y_list_fft).apply(lambda x: x.std())
        X_test['z_std_fft'] = pd.Series(z_list_fft).apply(lambda x: x.std())

        # FFT avg absolute diff
        X_test['x_aad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_test['y_aad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_test['z_aad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

        # FFT min
        X_test['x_min_fft'] = pd.Series(x_list_fft).apply(lambda x: x.min())
        X_test['y_min_fft'] = pd.Series(y_list_fft).apply(lambda x: x.min())
        X_test['z_min_fft'] = pd.Series(z_list_fft).apply(lambda x: x.min())

        # FFT max
        X_test['x_max_fft'] = pd.Series(x_list_fft).apply(lambda x: x.max())
        X_test['y_max_fft'] = pd.Series(y_list_fft).apply(lambda x: x.max())
        X_test['z_max_fft'] = pd.Series(z_list_fft).apply(lambda x: x.max())

        # FFT max-min diff
        X_test['x_maxmin_diff_fft'] = X_test['x_max_fft'] - X_test['x_min_fft']
        X_test['y_maxmin_diff_fft'] = X_test['y_max_fft'] - X_test['y_min_fft']
        X_test['z_maxmin_diff_fft'] = X_test['z_max_fft'] - X_test['z_min_fft']

        # FFT median
        X_test['x_median_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(x))
        X_test['y_median_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(x))
        X_test['z_median_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(x))

        # FFT median abs dev 
        X_test['x_mad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_test['y_mad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_test['z_mad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

        # FFT Interquartile range
        X_test['x_IQR_fft'] = pd.Series(x_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_test['y_IQR_fft'] = pd.Series(y_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_test['z_IQR_fft'] = pd.Series(z_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # FFT values above mean
        X_test['x_above_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x > x.mean()))
        X_test['y_above_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x > x.mean()))
        X_test['z_above_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x > x.mean()))

        # FFT number of peaks
        X_test['x_peak_count_fft'] = pd.Series(x_list_fft).apply(lambda x: len(find_peaks(x)[0]))
        X_test['y_peak_count_fft'] = pd.Series(y_list_fft).apply(lambda x: len(find_peaks(x)[0]))
        X_test['z_peak_count_fft'] = pd.Series(z_list_fft).apply(lambda x: len(find_peaks(x)[0]))

        # FFT skewness
        X_test['x_skewness_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.skew(x))
        X_test['y_skewness_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.skew(x))
        X_test['z_skewness_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.skew(x))

        # FFT kurtosis
        X_test['x_kurtosis_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.kurtosis(x))
        X_test['y_kurtosis_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.kurtosis(x))
        X_test['z_kurtosis_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.kurtosis(x))

        # FFT energy
        X_test['x_energy_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x**2)/50)
        X_test['y_energy_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x**2)/50)
        X_test['z_energy_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x**2/50))

        # FFT avg resultant
        X_test['avg_result_accl_fft'] = [i.mean() for i in ((pd.Series(x_list_fft)**2 + pd.Series(y_list_fft)**2 + pd.Series(z_list_fft)**2)**0.5)]

        # FFT Signal magnitude area
        X_test['sma_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(abs(x)/50)) + pd.Series(y_list_fft).apply(lambda x: np.sum(abs(x)/50)) \
                             + pd.Series(z_list_fft).apply(lambda x: np.sum(abs(x)/50))

        # Max Indices and Min indices 

        # index of max value in time domain
        X_test['x_argmax'] = pd.Series(x_list).apply(lambda x: np.argmax(x))
        X_test['y_argmax'] = pd.Series(y_list).apply(lambda x: np.argmax(x))
        X_test['z_argmax'] = pd.Series(z_list).apply(lambda x: np.argmax(x))

        # index of min value in time domain
        X_test['x_argmin'] = pd.Series(x_list).apply(lambda x: np.argmin(x))
        X_test['y_argmin'] = pd.Series(y_list).apply(lambda x: np.argmin(x))
        X_test['z_argmin'] = pd.Series(z_list).apply(lambda x: np.argmin(x))

        # absolute difference between above indices
        X_test['x_arg_diff'] = abs(X_test['x_argmax'] - X_test['x_argmin'])
        X_test['y_arg_diff'] = abs(X_test['y_argmax'] - X_test['y_argmin'])
        X_test['z_arg_diff'] = abs(X_test['z_argmax'] - X_test['z_argmin'])

        # index of max value in frequency domain
        X_test['x_argmax_fft'] = pd.Series(x_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
        X_test['y_argmax_fft'] = pd.Series(y_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
        X_test['z_argmax_fft'] = pd.Series(z_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))

        # index of min value in frequency domain
        X_test['x_argmin_fft'] = pd.Series(x_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
        X_test['y_argmin_fft'] = pd.Series(y_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
        X_test['z_argmin_fft'] = pd.Series(z_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))

        # absolute difference between above indices
        X_test['x_arg_diff_fft'] = abs(X_test['x_argmax_fft'] - X_test['x_argmin_fft'])
        X_test['y_arg_diff_fft'] = abs(X_test['y_argmax_fft'] - X_test['y_argmin_fft'])
        X_test['z_arg_diff_fft'] = abs(X_test['z_argmax_fft'] - X_test['z_argmin_fft'])

        return X_test
        
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
            fig.update_layout(legend={'title':'ACTIVIDAD','orientation':'h','y':1.2,'itemclick':'toggle'})


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
        #columna = st.selectbox('Escoge una columna',columnas)

        # Crea el objeto "modelo"
        modelo = Modelo(df)

        # Slider para seleccionar la frecuencia de corte
        fs = modelo.fs
   
        
        
        
if file is not None:
    
    st.write(os.listdir())
    lr_model = joblib.load('../models/logistic-model-ar-rev02.joblib')
    lr_scaler = joblib.load('../models/logistic-model-ar-scaler-rev02.joblib')
    #lr_function = joblib.load('../models/logistic-model-get-params-function.joblib')
    get_x_params = pickle.load(open('../models/logistic-model-get-params-function.joblib', 'rb'))
    
    #X_test = modelo.get_x_params(df)
    X_test = get_x_params(df)
    X_scaled = lr_scaler.transform(X_test)
    Y_pred = lr_model.predict(X_scaled)
    
    idx = 0
    for i in range(0, df.shape[0] - 250, 125):
        df.loc[i:i+125,'ACTIVIDAD'] = Y_pred[idx]
        idx = idx + 1
    
 

    # Para graficar los datos sin filtrar
    st.subheader('Datos sin Analizar')
    st.plotly_chart(modelo.graficar_columna('y', actividad=None),use_container_width=True)
    
    # Para graficar los datos sin filtrar
    st.subheader('Datos sin Analizar')
    st.plotly_chart(modelo.graficar_columna('y', actividad='TODAS'),use_container_width=True)
