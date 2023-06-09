import pandas as pd

def test_vectorization():
  def test_single_vector():
    test_value = procesar_texto('This is a test in english')[0]
    if len(test_value) == 768:
      print('Test_single_vector passed')
    else:
      print('Test_single_vector failed')
      print(test_value)

  def test_single_vector_spanish():
    test_value = procesar_texto('Esto es una prueba en español')[0]
    if len(test_value) == 768:
      print('Test_single_vector_spanish passed')
    else:
      print('Test_single_vector_spanish failed')
      print(test_value)

  def test_list_of_vectors():
    df = pd.DataFrame()
    df['Abstract'] = ['This is the article number 1', 'This is the article number 2',
                      'Este es el artículo numero 3']
    vectores = base_de_datos_de_vectores(df)
    if len(vectores.iloc[0]['Vector'][0]) == 768:
      print('Test_list_of_vectors passed')
    else:
      print('Test_list_of_vectors failed')
      print(len(vectores.iloc[0]['Vector'][0]))
    
  test_single_vector()
  test_single_vector_spanish()
  test_list_of_vectors()
test_vectorization()
