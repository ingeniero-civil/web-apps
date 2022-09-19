# Graficar datos georreferenciados (lat, lon) para archivos generados por la aplicación "AndroSensor", de Android
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from geopy.distance import geodesic


########## MODELO ##########
class Modelo:
    def __init__(self, df):
        self.df = df
        
    def plot_map(self, df, col_lat, col_lon, col_time=None):
        x1, x2 = df[col_lon].max(),df[col_lon].min()
        y1, y2 = df[col_lat].max(),df[col_lat].min()

        max_bound = max(abs(x1-x2), abs(y1-y2)) * 111
        zoom = 14 - np.log(max_bound)
        
        lat = df[col_lat].values
        lon = df[col_lon].values
        n = len(lat)
        d = [0]
        for i in range(1,n):
            p1 = (lat[i],lon[i])
            p2 = (lat[i-1],lon[i-1])
            d.append(geodesic(p1,p2).m)
            
        df['D'] = d
        
        if col_time is not None:
            try:
                t = pd.to_datetime(df[col_time],format='%Y-%m-%d %H:%M:%S:%f')
                dt = t.diff().dt.total_seconds()
                df['DT'] = dt
                df['V (KPH)'] = df['D'].div(df['DT']).mul(18/5)
                
                fig = px.scatter_mapbox(df,
                        lon='LOCATION Longitude : ',
                        lat='LOCATION Latitude : ',
                        mapbox_style="open-street-map",
                        zoom=zoom,
                        color='V (KPH)',
                        height=800,
                        range_color=[df['V (KPH)'].quantile(0.05),df['V (KPH)'].quantile(0.95)]
                       )
            except:
                fig = px.scatter_mapbox(df,
                                    lon=col_lon,
                                    lat=col_lat,
                                    mapbox_style="open-street-map",
                                    zoom=zoom,
                                    height=600
                                   )
        else:
            fig = px.scatter_mapbox(df,
                                    lon=col_lon,
                                    lat=col_lat,
                                    mapbox_style="open-street-map",
                                    zoom=zoom,
                                    height=600
                                   )
        return fig
    
########## VISTA ##########
st.set_page_config(layout = 'wide',page_title='Gráfico de Puntos Georreferenciados', page_icon="🗺️")
st.title('Gráfico de Puntos Georreferenciados')

with st.sidebar:
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
    idx_lat, idx_lon, idx_time = None, None, None
    for idx, columna in enumerate(columnas):
        if 'Lat' in columna:
            idx_lat = idx
        elif 'Lon' in columna:
            idx_lon = idx
        elif 'YYYY' in columna:
            idx_time = idx
            
    if idx_lat is not None:
        col_lat = st.sidebar.selectbox('Escoge la columna con los datos de latitud',columnas,index=idx_lat)
    else:
        col_lat = st.sidebar.selectbox('Escoge la columna con los datos de latitud',columnas)
        
    if idx_lon is not None:
        col_lon = st.sidebar.selectbox('Escoge la columna con los datos de longitud',columnas, index=idx_lon)
    else:
        col_lon = st.sidebar.selectbox('Escoge la columna con los datos de longitud',columnas)
        
    if idx_time is not None:
        col_time = st.sidebar.selectbox('Escoge la columna con los datos de tiempo',columnas, index=idx_time)
    else:
        col_time = st.sidebar.selectbox('Escoge la columna con los datos de tiempo',columnas)
    
    # Crea el objeto "modelo"
    modelo = Modelo(df)
    
    st.plotly_chart(modelo.plot_map(df, col_lat, col_lon, col_time),use_container_width=True)
