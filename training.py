## Librerías:
# Para 'intents.json'
import json # Para el formato json
import codecs # Lectura de caracteres en español
# Procesamiento de Lenguaje Natural
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import pickle # Guardar archivos
import numpy as np
import nltk # Natural Language ToolKit: Tokenizar
from recursos import lemmatizer, normalize_tilde

# Redes Neuronales:
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

## 1. Cargar la información del archivo json

def load_training_data(json_file = "intents.json"):
    # Cargar archivo json:
    intents = json.loads(codecs.open(json_file, encoding='utf-8').read())
    
    words = [] # Palabras individuales usadas
    tags = [] # Etiquetas de intención
    ignore_letters = ['¿', '?', '.', ',', ';', '¡', '!', '(', ')']
    documents = [] # Lista de tuplas ([Lista de palabras], etiqueta asociada de intención)
    for intent in intents['intents']:
        # Por cada intención:
        for pattern in intent['patterns']:
            # Tokenizar el patrón de palabras que usaría el usuario y agregar las palabras a 'words'.
            pattern = lemmatizer(pattern) #........ Para separar '¿__w1__w2' y pasar palabras a su forma base.
            pattern = normalize_tilde(pattern) #... Quitas las tildes.
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Agregar a 'documents' la tupla (lista de palabras, etiqueta de la intención)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in tags:
                tags.append(intent['tag'])
            # Fin del segundo bucle
        # Fin del primer bucle
    words = [lemmatizer(word.lower()) for word in words if word not in ignore_letters]
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
print("Palabras detectadas:", len(words))
print("Intenciones detectadas:", len(tags))

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

train_x = list(training[:,0]) # Entradas
train_y = list(training[:,1]) # Salidas

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = SGD(lr = 0.01, decay =1e-6, momentum=0.9, nesterov = True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose = 1)
model.save('chatbot_model.model')
print("Red Neuronal Guardada")