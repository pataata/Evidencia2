# file: import_text.py

import pandas as pd
import numpy as np
import os

def importar_datos(directorio):
  def importar_texto(nombre_archivo):
    with open(nombre_archivo) as archivo:
        return archivo.read()

  textos = []
  filenames = []
  for subdir, dirs, archivos in os.walk(directorio):
    for archivo in archivos:
      textos.append(importar_texto(f'{directorio}/{archivo}'))
      filenames.append(archivo)
  df = pd.DataFrame()
  df['Texto'] = textos
  df['Index'] = [i for i in range(len(textos))]
  df['Filename'] = filenames
  return df

def importar_csv(archivo, vectores=False):
    if vectores:
      df = pd.read_csv(archivo, sep="|")
      vectores = []
      for i in range(len(df)):
        row = df.iloc[i]
        vec = row['Vector']
        vec = vec.split(';')
        new_vec = []
        for num in vec:
          if num != '':
            new_vec.append(float(num))
        vectores.append(np.array([new_vec]))
      df['Vector'] = vectores
      return df
    else:
      return pd.read_csv(archivo, sep="|", names=['Abstract','Plagiarism','Filename'])

def serializar_datos(nombre, dataframe):
  df = dataframe
  i = 0
  vectores = []
  for i in range(len(df)):
    old = df.iloc[i]['Vector'][0]
    vec = ''
    for num in old:
      vec += str(num)
      vec += ';'
    vectores.append(vec)
  df['Vector'] = vectores

  df.to_csv(sep='|', path_or_buf=nombre)