# filename: main.py
import pandas as pd
from import_text import *
from vectorize_text import *
from metrics import *

# filename: main.py

UMBRAL_PLAGIO = .931
ANALIZAR_MODELO = True
VECTORES_CALCULADOS = True

## main
if VECTORES_CALCULADOS:
  datos_vectores = importar_csv('articulos_procesados.csv', True)
  base_datos = datos_vectores[datos_vectores['Plagiarism'] == 0]
else:
  datos = importar_csv('articulos.csv')

if ANALIZAR_MODELO:
  #Vectorizar datos
  try: datos_vectores
  except NameError:
    datos_vectores = base_de_datos_de_vectores(datos)
  base_datos = datos_vectores[datos_vectores['Plagiarism'] == 0]
  #Obtener las similitudes
  similitudes = []
  indice_similar = []
  for index, row in datos_vectores.iterrows():
      base_datos_limpia = base_datos[base_datos['Index'] != row['Index']]
      result = analisis_plagio(row['Abstract'], base_datos_limpia, UMBRAL_PLAGIO, calcular_vector=False, vector=row['Vector'])
      similitudes.append(result['similitud'])
      indice_similar.append(result['indice similar'])
  datos_procesados = datos_vectores
  datos_procesados['Similitud'] = similitudes
  datos_procesados['Indice similar'] = indice_similar

  #Análisis de los datos
  stats, tabla_resultados = test_precision(datos_vectores, UMBRAL_PLAGIO)
  print(stats)
  tabla_resultados
else:
  try: base_datos
  except:
    datos = datos[datos['Plagiarism'] == 0]
    base_datos = base_de_datos_de_vectores(datos)
  directorio = 'docmentos-sospechosos'
  resultados_deteccion = analizar_archivos(directorio, base_datos)
  resultados_deteccion = resultados_deteccion.drop(['Index'], axis=1)
print(tabla_resultados)