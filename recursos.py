## Librerías:
# Procesamiento de Lenguaje Natural
from re import A
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import numpy as np
import pandas as pd
import nltk # Natural Language ToolKit: Tokenizar
import pickle

# Preparación
nltk.download('punkt')
nlp = spacy.load('es_core_news_sm') # Procesamiento de Lenguaje con base al español
words = pickle.load(open('words.pkl','rb'))

# Lematizador que acepta texto con Spacy español
def lemmatizer(text):
    """
    - text: Texto de entrada
    Retorna string de la misma oración con cada palabra lematizada
    """
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])

def analyze(text):
    """
    Análisis (sintaxis) de la oración para determinar el sujeto, objeto, 'raíz', etc.
    de una oración (string)
    """
    doc = nlp(text)
    sentence = next(doc.sents) 
    for word in sentence:
        print("%s:%s" % (word,word.dep_))

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

def make_keywords(text):
    """
    Recibe texto y crea keywords a partir del atributo word.dep_ de spacy
    """
    doc = nlp(text)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") or (tok.dep_ == "ROOT") or (tok.dep_ == "flat") or (tok.dep_ == "dobj")]
    return sub_toks

"""
print("GO.")
while True:
    text = input("")
    if text == "SALIR":
        break
    text = filter(text, words) # Arreglo (oración tokenizada y filtrada)
    text = ' '.join([word for word in text]) # Juntar los tokens en un solo string 
    # analyze(text)
    tokens = make_keywords(text)
    print(tokens)
"""

# SIMULANDO CON DATAFRAMES COMO BASE DE DATOS (?)

fun_facts = pd.read_csv('data_prueba/fun_facts.csv', sep='|')
touristic_place = pd.read_csv('data_prueba/touristic_place.csv', sep='|')


def fake_query(keywords, query_from: str, column_target: str):
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
    
    touristic_places = bs_dataframe[column_place_name] # Arreglo de los nombres de los lugares dentro de la base de datos
    touristic_places = set(touristic_places) # Conjunto para evitar problemas
    touristic_places = [place.lower() for place in touristic_places] # Convertir a Lista, en minúscula para que sea par a keywords
    list_i = [0 for _ in range(len(touristic_places))] # Contador para ver qué instancia (lugar) es el más adecuado según keywords
    for i in range(len(touristic_places)):
        for word in keywords:
            if word in touristic_places[i]:
                list_i[i] = list_i[i] + 1
    i = list_i.index(np.max(list_i)) # El índice del lugar que más coincidencias tiene con mis keywords.
    aux = bs_dataframe[bs_dataframe[column_place_name] == touristic_places[i].upper()]
    aux = aux[column_target]
    aux = [a for a in aux]
    return aux

#print(fun_facts[fun_facts['touristic_place_id'] == 'MUSEO DEL BANCO CENTRAL DE RESERVA DEL PERÚ'])