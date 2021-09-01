## Librerías:
# Para 'intents.json'
import json # Para el formato json
import codecs # Lectura de caracteres en español
# Procesamiento de Lenguaje Natural
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import pickle
import numpy as np
import nltk # Natural Language ToolKit: Tokenizar
# Redes Neuronales:
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from recursos import lemmatizer

"""
# Preparación
nltk.download('punkt')

# Lematizador que acepta texto con Spacy español
nlp = spacy.load('es_core_news_sm')
def lemmatizer(text):
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])
"""

# 1. Cargar la información del archivo json
def load_training_data(json_file = "intents.json"):
    # Cargar archivo json:
    intents = json.loads(codecs.open(json_file, encoding='utf-8').read())
    print("Archivo json cargado")
    
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
    return words, tags, documents

words, tags, documents = load_training_data()

# Guardarlos
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

# 2. Bag of Words: Formar matriz Training 

training = []
output_empty = [0] * len(tags)

for document in documents:
    bag = [] # bag of words
    word_patterns = document[0]
    word_patterns = [lemmatizer(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word.lower() in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[tags.index(document[1])] = 1
    training.append([bag, output_row])


"""
tags = [Saludos, Trivia, ...]

Oracion         Etiqueta
[0,0,0,1,0]     [0,1]
"""

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

print("Formada matriz train: (%s,%s)"%(len(training), len(train_y[0])))

# 3. Red Neuronal

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