# filename: metrics.py
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

def analisis_plagio(texto, base_datos, umbral_plagio, top_N=3, calcular_vector=True, vector=None):
  if calcular_vector:
    texto_traducido = revisar_idioma_texto(texto)

    # Obtener embedding del texto
    vector = procesar_texto(texto_traducido)

  else:
    vector = np.array(vector)
    vector = vector.reshape(1, -1)

  # Crear tabla de resultados
  copia_base = base_datos
  copia_base["Similitud"] = copia_base["Vector"].apply(lambda x:
                                          cosine_similarity(vector, x)[0][0])
  resultados = copia_base.sort_values(by='Similitud',
                                      ascending=False)[0:top_N+1]
  score = resultados.iloc[0]['Similitud']
  articulo_mas_similar = resultados.iloc[0]['Abstract']
  es_plagio = 1 if score >= umbral_plagio else 0

  return {'similitud': score,
          'es_plagio': es_plagio,
          'articulo evaluado': texto,
          'articulo similar': articulo_mas_similar,
          'nombre similar': resultados.iloc[0]['Filename'],
          'indice similar': resultados.iloc[0]['Index'],
          }

def test_precision(dataset, umbral):
  falsos_positivos = 0
  falsos_negativos = 0
  reu_file = []
  predicted_y = []
  outliers = 0

  def actualizar_datos(reu, pred):
    reu_file.append(reu)
    predicted_y.append(pred)

  for index, row in dataset.iterrows():
    sim = row['Similitud']
    if row['Plagiarism'] == 0:
        if(sim >= .96):
          outliers += 1
          actualizar_datos(dataset.iloc[row['Indice similar']]['Filename'], 'X')
        else:
          if sim >= umbral:
            falsos_positivos += 1
            actualizar_datos(dataset.iloc[row['Indice similar']]['Filename'], '1')
          else:
            actualizar_datos(dataset.iloc[row['Indice similar']]['Filename'],'0')
    else:
      if sim < umbral:
        falsos_negativos += 1
        actualizar_datos(dataset.iloc[row['Indice similar']]['Filename'], '0')
      else:
        actualizar_datos(dataset.iloc[row['Indice similar']]['Filename'], '1')
  real_len = len(dataset) - outliers
  stats = {
      'Numero de muestras' : real_len,
      'falsos positivos' : falsos_positivos,
      'falsos negativos' : falsos_negativos,
      'Accuracy' : ((real_len - (falsos_positivos + falsos_negativos))/real_len)
      }
  tabla_resultados = pd.DataFrame()
  tabla_resultados['Texto original'] = dataset['Filename']
  tabla_resultados['Valor esperado'] = dataset['Plagiarism']
  tabla_resultados['Valor predecido'] = predicted_y
  tabla_resultados['Similitud'] = dataset['Similitud']
  tabla_resultados['Texto Reutilizado'] = reu_file
  return stats, tabla_resultados

def analizar_archivos(directorio, base_datos):
  df = importar_datos(directorio)
  plagio = []
  nombre_plg = []
  texto_plg = []
  similitud = []
  for index, row in df.iterrows():
    resultado = analisis_plagio(row['Texto'], base_datos, UMBRAL_PLAGIO)
    plagio.append(resultado['es_plagio'])
    nombre_plg.append(resultado['nombre similar'])
    texto_plg.append(resultado['articulo similar'])
    similitud.append(resultado['similitud'])
  df['Plagiarism'] = plagio
  df['Similitud'] = similitud
  df['Artículo más similar'] = nombre_plg
  df['Abstract similar'] = texto_plg
  return df

def comparar_dos_textos(texto1, texto2):
  print(cosine_similarity(procesar_texto(texto1),procesar_texto(texto2))[0][0])