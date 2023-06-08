import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

########## MODELO ##########
class Modelo:
    def __init__(self, df):
        self.df = df
        
    def plot_map(self, df, col_lat, col_lon):
        x1, x2 = df[col_lon].max(),df[col_lon].min()
        y1, y2 = df[col_lat].max(),df[col_lat].min()

        max_bound = max(abs(x1-x2), abs(y1-y2)) * 111
        zoom = 14 - np.log(max_bound)

        fig = px.scatter_mapbox(df,
                                lon=col_lon,
                                lat=col_lat,
                                mapbox_style="open-street-map",
                                zoom=zoom
                               )
        return fig
    
########## VISTA ##########
st.set_page_config(layout = 'wide')
st.title('Procesamiento de Datos de INEI')

st.header('Subir archivo')
file = st.file_uploader('Archivo en formato XLSX',type='xlsx')
if file is not None:
    # Leer el archivo subido a la página
    # Rutina para ignorar la primera fila si contiene la palabra 'sep'
    df0 = pd.read_excel(file,header=5,names=np.arange(0,5))
    
    a = df0[1].str.contains('AREA')
    l1 = a[a==True].index.to_list()

    a = df0[1].str.contains('RESUMEN')
    idx = a[a == True].index[0]
    l2 = l1[1:]
    l2.append(idx)
    
    n = len(l1)
    lista = []
    for e in range(n):
        idx = l1[e]
        area = df0.iloc[idx,1][7:]
        lugar = df0.iloc[idx,2].split(', ')

        codManzana = lugar[0].strip()
        lugar2 = lugar[1].upper().split(',') #['LIMA', 'LIMA', 'COMAS', 'CENTRO POBLADO: LA LIBERTAD', 'MZA: 001A']
        departamento, provincia, distrito, centroPoblado, mza = lugar2[0],lugar2[1],lugar2[2],lugar2[3][16:],lugar2[4][5:]

        i, f = l1[e]+3, l2[e]-1
        #df = pd.DataFrame(df0.iloc[i:f,1:].values).set_index(0)
        d = {'AREA':area,'CODIGO':codManzana,'MANZANA':mza,'DEPARTAMENTO':departamento,'PROVINCIA':provincia,'DISTRITO':distrito}
        d.update(df0.iloc[i:f,1:].set_index([1])[2].to_dict())
        lista.append(d)
        
    df = pd.DataFrame(lista).fillna(0)
    df['AREA'] = df['AREA'].astype(str)
    df['CODIGO'] = df['CODIGO'].astype(str)
    df['MANZANA'] = df['MANZANA'].astype(str)
        
    # Botón para descargar los datos filtrados
    st.subheader('Descargar Datos Procesados')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos como CSV",
        data=csv,
        file_name='inei_procesado.csv',
        mime='text/csv')
