# file: vectorize_text.py

import numpy as np
import torch
from keras.utils import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
from translate import *

# cargar modelo de bert y el tokenizer
nombre_modelo = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(nombre_modelo,
                                         do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(nombre_modelo,
                                                         output_attentions=False,
                                                         output_hidden_states=True)

def obtener_vector_de_texto(tokenizer, model, texto, max_length = 512):
  texto = revisar_idioma_texto(texto)

  ids_texto = tokenizer.encode(
                      text = texto,
                      max_length = max_length,
                      add_special_tokens = True                 
  )
  
  input_ids = pad_sequences([ids_texto],maxlen=max_length, dtype="long",
                            truncating="post", padding="post")[0]
  # Crear mascara de atencion   
  attention_mask = [int(i>0) for i in input_ids]

  # Convertir a tensores
  input_ids = torch.tensor(input_ids)
  attention_mask = torch.tensor(attention_mask)
  # Agregar Dimension adicional para el batch
  input_ids = input_ids.unsqueeze(0)
  attention_mask = attention_mask.unsqueeze(0)

  # Modo evaluación del modelo, forward propagation
  model.eval()
  # Proporcinar el texto a Bert, y recolectar los valores de las 12 capas
  with torch.no_grad():       
    logits, encoded_layers = model(
                                input_ids = input_ids,
                                token_type_ids = None,
                                attention_mask = attention_mask,
                                return_dict=False)

  layer_i = 12 # Última capa de Bert, antes del clasificador.
  batch_i = 0 # Numero de Batches (1).
  token_i = 0 # El primer token, corresponde al [CLS]
    
  # Extraer el vector.
  vector = encoded_layers[layer_i][batch_i][token_i]
  # Mover al CPU y convertir a numpy ndarray.
  vector = vector.detach().cpu().numpy()
  return(vector)

def base_de_datos_de_vectores(datos):
  # La lista de los vectores
  vectores = []

  # Obtener la información a procesar
  textos = datos['Abstract']

  # Obtener los embeddings de todos los textos
  for texto in textos:
    vector = obtener_vector_de_texto(tokenizer, model, texto)
    vectores.append(vector)

  # Agregar a la serie de datos y dar formato
  datos['Vector'] = vectores
  datos['Vector'] = datos['Vector'].apply(lambda emb: np.array(emb))
  datos['Vector'] = datos['Vector'].apply(lambda emb: emb.reshape(1, -1))
  datos['Index'] = [i for i in range(len(datos))]
  datos.set_index('Index')
  return datos

def procesar_texto(texto):
  """
  Obtener el embedding de un texto
  """
  vec_texto = obtener_vector_de_texto(tokenizer, model, texto)
  vec_texto = np.array(vec_texto)
  vec_texto = vec_texto.reshape(1, -1)
  return vec_texto