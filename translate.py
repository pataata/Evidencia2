# filename: translate.py

from langdetect import detect, DetectorFactory
from transformers import MarianTokenizer, MarianMTModel

DetectorFactory.seed = 0

def traducir_texto(texto, idioma_origen, idioma_destino='en'):
  # Obtener el modelo adecuado
  nombre_modelo = f"Helsinki-NLP/opus-mt-{idioma_origen}-{idioma_destino}"
  tokenizer = MarianTokenizer.from_pretrained(nombre_modelo)

 # Instanciar el modelo
  model = MarianMTModel.from_pretrained(nombre_modelo)
 
  # Traducción del texto
  texto_formato = ">>{}<< {}".format(idioma_origen, texto)[:512]
  traduccion = model.generate(**tokenizer([texto_formato], 
                               return_tensors="pt", padding=True))
  texto_traducido = [tokenizer.decode(t, skip_special_tokens=True) for t in traduccion][0]
 
  return texto_traducido

def revisar_idioma_texto(texto):
  idioma = detect(texto)
  lista_idiomas = ['es','de', 'fr', 'el', 'ja']
  resultado = ""
  if(idioma == 'en'):
    resultado = texto
  elif(idioma not in lista_idiomas):
    print('Idioma no detectado, los resultados se verán afectados')
    resultado = texto
  else:
    # Translate in English
    resultado = traducir_texto(texto, idioma)
  return resultado