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
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") or (tok.dep_ == "ROOT") or (tok.dep_ == "flat") ]
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