## Librerías:
# Procesamiento de Lenguaje Natural
from re import A
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import numpy as np
import pandas as pd
import nltk # Natural Language ToolKit: Tokenizar
import pickle

import geocoder

# Preparación
nltk.download('punkt')
nlp = spacy.load('es_core_news_sm') # Procesamiento de Lenguaje con base al español
words = pickle.load(open('words.pkl','rb'))

# Lematizador que acepta texto con Spacy español
def lemmatizer(text):
    """
    - text: Texto de entrada
    Retorna string de la misma oración con cada palabra lematizada
    E.g: "Buenos días" -> "Buen día"
    """
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])

# Nota: Usar para inspeccionar la estructura de una oración de entrada. Actualmente no es llamada por ningún método.
def analyze(text):
    """
    Análisis (sintaxis) de la oración para determinar el sujeto, objeto, 'raíz', etc.
    de una oración (string)
    NOTA: Es simplemente un print de cada palabra en una oración, con su atributo
    """
    doc = nlp(text)
    sentence = next(doc.sents) 
    for word in sentence:
        print("%s:%s" % (word,word.dep_))

# Filtro: Recibe una oración (text), a la que se le quitan las palabras baneadas en (list_words)
def filter(text, list_words):
    """
    Filtra las palabras de una oración, lematizando y quitando aquellas que sean parte del arreglo global words.
    - text: Oración de entrada
    - list_words: Lista de palabras que se van a eliminar en caso de ocurrencia.
    """
    ignore_letters = ['¿', '?', '.', ',', ';', '¡', '!', '(', ')']
    text = lemmatizer(text) #... 1. Lematizar
    tokens = nltk.word_tokenize(text) #... 2. Tokenizar para tener una lista de palabras
    bag = []
    for word in tokens:
        bag if word.lower() in list_words or word in ignore_letters else bag.append(word)
    return bag

# Quita las tildes con un simple LUT (Tal vez puede mejorarse)
def normalize_tilde(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

# A partir del análisis sintáctico (librería spacy), rescata las más importantes palabras en una oración (text)
def make_keywords(text):
    """
    Recibe texto y crea keywords a partir del atributo word.dep_ de spacy
    - text: Oración de entrada. (e.g: "Museo de arte de Lima")
    Retorna
    - sub_toks: Lista de palabras clave. (e.g: ["Museo", "Arte", "Lima"] )
    """
    doc = nlp(text)
    # print("Recibí: ", text)
    sub_toks = [normalize_tilde(str(tok)) for tok in doc if (tok.dep_ == "nsubj") or (tok.dep_ == "ROOT") or (tok.dep_ == "flat") or (tok.dep_ == "dobj") or (tok.dep_ == "amod") or (tok.dep_ == "appos")]
    return sub_toks

#================= ESTO ES LO 'SIMULADO' ==============

fun_facts = pd.read_csv('data_prueba/fun_facts.csv', sep='|')
touristic_place = pd.read_csv('data_prueba/touristic_place.csv', sep='|')

# OJO: Usa el par (latitud, longitud) de quien esté ejecutando esto.
# TO DO: Investigar en la documentación de geocoder para la comunicación frontend-backend
def get_user_location():
    """
    Propuesta: Trabajar con GeoCoder para tener la ubicación en formato
    Retorna: [Latitud, Longitud] (e.g: lista de [-12.0432, -77.0282]), como np.array para hacer distancia euclidiana
    """
    g = geocoder.ip('me')
    return np.array(g.latlng)

"""
Propuesta: Mezclar fake_query con query_near:
* Añadir al DataFrame una columna 'score'.
* score se calcula a partir de lambdas {1: Coincidencias entre el nombre y keywords. 2: Distancia Euclidiana}
* Ordenar de mayor a menor score.
* Retornar el mayor o mayores instancias según dicho orden.
"""

def fake_query(keywords, query_from: str, column_target: str, place_context: str):
    """
    keywords: Lista de keywords del lugar turístico que se quiere buscar
    query_from: Indica de qué csv (tabla) se va a extraer la información
    column_target: El nombre de la columna objetivo dentro de la tabla (lo que se va a retornar)
    """

    # 1. Preparar Variables
    bs_dataframe = touristic_place # La tabla de la que se va a consultar
    column_place_name = "name" # El nombre de la columna que guarda los nombres de los lugares
    if query_from == "fun_facts":
        bs_dataframe = fun_facts
        column_place_name = "touristic_place_id"
    
    if place_context == " " or len(keywords) > 0:
        # Si NO hay contexto
        # 2. Buscar las instancias de la tabla que contengan las keywords:
        touristic_places = bs_dataframe[column_place_name] # Arreglo de los nombres de los lugares dentro de la base de datos
        touristic_places = set(touristic_places) # Conjunto para evitar problemas
        touristic_places = [normalize_tilde(place) for place in touristic_places] # Convertir a Lista, Sin tildes para la búsqueda por keywords
        list_i = [0 for _ in range(len(touristic_places))] # Contador para ver qué instancia (lugar) es el más adecuado según keywords
        for i in range(len(touristic_places)):
            for word in keywords:
                if word.upper() in touristic_places[i]:
                    list_i[i] = list_i[i] + 1
        aux = np.max(list_i) # Máximo valor en el contador
        if aux == 0:
            return list([]), place_context # Si hay 0 resultados, regresa
        i = list_i.index(aux) # El índice del lugar que más coincidencias tiene con mis keywords.
        place_context = touristic_places[i].upper()
        print("> Lugar: ", place_context)
    
    print("... Contexto:", place_context)
    aux = bs_dataframe[bs_dataframe[column_place_name] == place_context]
    aux = aux[column_target]
    aux = [a for a in aux]
    return aux, place_context

# AHH
def query_near():
    longitudes = touristic_place['longitude']
    latitudes = touristic_place['latitude']
    distancia = []
    for i in range(len(longitudes)):
        dist = np.linalg.norm(get_user_location()-np.array([longitudes[i], latitudes[i]]))
        distancia.append(dist)
    touristic_place["Distancia"] = distancia # Añade columna de distancia euclidiana
    aux = touristic_place.sort_values(by=['Distancia'])
    # Retorna el más cercano
    return aux['name'][0] 

#print(fake_query(["Perú", "Museo", "Banco", "Central", "Reserva"], "fun_facts", "fact", " "))

#print(fun_facts[fun_facts['touristic_place_id'] == 'MUSEO DEL BANCO CENTRAL DE RESERVA DEL PERÚ'])