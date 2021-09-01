## Librerías:
# Para 'intents.json'
import json # Para el formato json
import codecs # Lectura de caracteres en español
# Procesamiento de Lenguaje Natural
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import pickle # Guardar archivos
import numpy as np
import nltk # Natural Language ToolKit: Tokenizar
# Preparación de la matriz de entrenamiento:
import random

from recursos import lemmatizer


## 1. Cargar la información del archivo json

def load_training_data(json_file = "intents.json"):
    # Cargar archivo json:
    intents = json.loads(codecs.open(json_file, encoding='utf-8').read())
    
    words = [] # Palabras individuales usadas
    tags = [] # Etiquetas de intención
    ignore_letters = ['¿', '?', '.', '!', '(', ')']
    documents = [] # Lista de tuplas ([Lista de palabras], etiqueta asociada de intención)
    for intent in intents['intents']:
        # Por cada intención:
        for pattern in intent['patterns']:
            # Tokenizar el patrón de palabras que usaría el usuario y agregar las palabras a 'words'.
            pattern = lemmatizer(pattern) # Para separar '¿:::'
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Agregar a 'documents' la tupla (lista de palabras, etiqueta de la intención)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in tags:
                tags.append(intent['tag'])
            # Fin del segundo bucle
        # Fin del primer bucle
    words = [lemmatizer(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    tags = sorted(set(tags))
    """
    Ejemplo:
    - words = ['Buen', 'Hola', 'día', ...]
    - tags = ['saludos', 'opciones', 'consulta_lugar']
    - documents = [(['Buen', 'día'], 'saludos'), (['Hola'], 'saludos'), ([...], 'opciones'), ...]
    """
    return words, tags, documents

words, tags, documents = load_training_data()
print("intents.json cargado")

# ... Guardarlos
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))
# ... 'documents' ayudará a crear la matriz de entrenamiento

# 2. Bag of Words: Formar matriz Training 

training = []
output_empty = [0] * len(tags)

for document in documents:
    # Por cada tupla ([consulta de palabras], tag)
    bag = [] # ... bag of words
    word_patterns = document[0] #... lista de palabras
    word_patterns = [lemmatizer(word.lower()) for word in word_patterns] #... lematizar las palabras en minúsculas
    for word in words:
        # Por cada palabra en el conjunto total de palabras encontradas en el archivo json.
        bag.append(1) if word.lower() in word_patterns else bag.append(0)
        # Si dicha palabra está entre el patrón de este documento: 1; else -> 0
    # - bag = [0,0,0,...,1], de len(bag) = len(words)
    
    output_row = list(output_empty) # Hacer una lista de 0, del tamaño de len(tags)
    output_row[tags.index(document[1])] = 1 # Cambia a '1' la celda de mismo índice que tag en tags.
    training.append([bag, output_row]) # Se agrega a la matriz de Training.

random.shuffle(training) # Mezclar Data de Entrenamiento
training = np.array(training)
print("Formada matriz training. Filas: %s, Etiquetas:%s)"%(len(training), len(training[0][1])))
pickle.dump(training, open('training.pkl', 'wb'))