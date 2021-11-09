## Librerías:
# Procesamiento de Lenguaje Natural
# from re import A
from nltk.util import pr
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import numpy as np
import pandas as pd
import nltk # Natural Language ToolKit: Tokenizar
import pickle

import geocoder

# Preparación
# nltk.download('punkt')
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
    sub_toks = [normalize_tilde(str(tok)) for tok in doc if (tok.dep_ == "nsubj") or (tok.dep_ == "ROOT") or (tok.dep_ == "flat") or (tok.dep_ == "dobj") or (tok.dep_ == "amod") or (tok.dep_ == "appos") or (tok.dep_ == "dep") or (tok.dep_ == "obj")]
    return sub_toks

#================= ESTO ES LO 'SIMULADO' ==============

# Tablas de la Base de Datos:
fun_facts = pd.read_csv('data_prueba/fun_facts.csv', sep='|')
touristic_place = pd.read_csv('data_prueba/touristic_place.csv', sep='|')
touristic_place_category = pd.read_csv('data_prueba/touristic_place_category.csv', sep='|')
url_images = pd.read_csv('data_prueba/url_images.csv', sep='|')
user_context = pd.read_csv('data_prueba/user_context.csv', sep='|')

# Nombres de todos los lugares existentes en la base de datos (Lista)
names = list(touristic_place_category['touristic_place_id'])
aux_names = [normalize_tilde(place) for place in names] # Convertir a Lista, Sin tildes para la búsqueda por keywords

# OJO: Usa el par (latitud, longitud) de quien esté ejecutando esto.
def get_user_location():
    """
    Propuesta: Trabajar con GeoCoder para tener la ubicación en formato
    Retorna: [Latitud, Longitud] (e.g: lista de [-12.0432, -77.0282]), como np.array para hacer distancia euclidiana
    """
    g = geocoder.ip('me')
    return np.array(g.latlng)

# Consulta del lugar más cercano (Retorna String del Nombre del lugar)
def query_near(user_location: np.array):
    """
    user_location: Coordenadas del usuario, en forma np.array([longitud, latitud])
    """
    longitudes = touristic_place['longitude']
    latitudes = touristic_place['latitude']
    distancia = []
    for i in range(len(longitudes)):
        dist = np.linalg.norm(user_location-np.array([longitudes[i], latitudes[i]]))
        distancia.append(dist)
    touristic_place["Distancia"] = distancia # Añade columna de distancia euclidiana
    aux = touristic_place.sort_values(by=['Distancia'])
    #... Retorna el más cercano
    return aux.values[0][0] # Asumimos que == [0]['name'], retorna el string con el nombre de lugar

# Selecciona solamente los nombres con mayor coincidencia (keywords / names)
def select_names(keywords: list, place_context: str):
    """
    * keywords: Lista de 'keywords' del lugar
    * place_context: Contexto (string del nombre del lugar)
    """
    keywords = [normalize_tilde(keyword) for keyword in keywords]
    # Si NO hay contexto o SÍ se han ingresado keywords
    if place_context == " " or len(keywords) > 0:
        # Buscar los nombres de los lugares que contengan las keywords:
        list_i = [(i, 0) for i in range(len(aux_names))] # Contador para ver qué instancia (lugar) es el más adecuado 
        #... según keywords. e.g: [(0, # de coincidencias), (1, #), (2,#)]
        # print("KEYWORDS:", keywords)
        for i in range(len(aux_names)):
            for word in keywords:
                if word.upper() in aux_names[i]:
                    # print("Se encontró %s en %s"%(word.upper(),aux_names[i]))
                    list_i[i] = (i, list_i[i][1] + 1) # (indice, numero de coincidencias)
        list_i.sort(key=lambda x: x[1], reverse=True) # Ordenar los nombres por mayor
                            #... coincidencias con las keywords
        max_coindicences = list_i[0][1]
        if max_coindicences == 0:
            # Si NO hay coincidencias
            return [place_context]
        res = []
        for i in range(len(list_i)):
            if list_i[i][1] < max_coindicences:
                return res
            res.append(names[list_i[i][0]])
    return [place_context]

def new_query(select_column: list, from_data: str, where_pairs: list):
    """
    * select_column: Análogo a 'SELECT' de SQL. Ingresa los nombres de las columnas (e.g: [str, str])
    * from_data: Análogo a 'FROM' de SQL. Ingresa el nombre de la tabla de la que se va a extraer la información
    * where_pairs: Lista de tuplas (columna, valor). Las condiciones que se desean cumplir 
                e.g: ((c_1 == v_1) and (c_2 == v_2) ...), en where_pairs = [(c_1,v_1), (c_2, v_2), ...]
    """
    # FROM
    bs_dataframe = touristic_place # La tabla de la que se va a consultar
    if from_data == "fun_facts":
        bs_dataframe = fun_facts
    elif from_data == "touristic_place_category":
        bs_dataframe = touristic_place_category
    elif from_data == "url_images":
        bs_dataframe = url_images
    elif from_data == "user_context":
        bs_dataframe = user_context
    # WHERE
    where_str = ""
    for i in range(len(where_pairs)):
        pair = where_pairs[i] # pair: (str_columna, valor)
        where_str = where_str + "%s == %s"%(pair[0], pair[1])
        if i < len(where_pairs) - 1:
            where_str = where_str + " and "
    aux = bs_dataframe.query(where_str)
    # SELECT
    return aux[select_column]

# TO DO: Terminar toda la función
def add_row(row: list, from_data: str, replace: bool = True, key_column: str = "id") -> None:
    """
    * row(s): Fila a agregar al dataset. Formato: [{v1: valor, v2: valor, ...}]
    * from_data: Nombre de la tabla sobre la cual se va a ejecutar este formato
    * replace: Indica si debería actualizar (reemplazar) la fila (1) en lugar de añadir
    * key_column: SOLO en caso de REPLACE. Nombre de la columna 'llave' con la que se define la fila a reemplazar
    """
    # FROM
    bs_dataframe = touristic_place # La tabla de la que se va a consultar
    if from_data == "fun_facts":
        bs_dataframe = fun_facts
    elif from_data == "touristic_place_category":
        bs_dataframe = touristic_place_category
    elif from_data == "url_images":
        bs_dataframe = url_images
    elif from_data == "user_context":
        bs_dataframe = user_context
    
    if replace:
        return
    bs_dataframe.append(row, ignore_index=True)
    return

#print(analyze("ñññ no preocupar palacio justicia"))
