## Librerías:
# Para 'intents.json'
import json # Para el formato json
import codecs # Lectura de caracteres en español
import pickle
# from google.protobuf import message
import nltk
import random
import numpy as np
from numpy.lib.function_base import place

from recursos import get_user_location, lemmatizer, filter, make_keywords, fake_query, query_near

from tensorflow.keras.models import load_model

# Cargar los archivos guardados por training.py
intents = json.loads(codecs.open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
model = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    """
    Limpiar oración de entrada (lematizar cada palabra)
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    Convertir la oración de entrada a bag_of_words
    """
    sentence_words = clean_up_sentence(sentence)
    # sentence_words: Arreglo de las palabras lematizadas
    bag = [0] * len(words)
    # bag: Arreglo de 0s
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                # Cambiar a '1' donde haya incidencia (bag of words)
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) # Representación bag of words de la oración de entrada (e.g: [0,0,0,...,1,0,1])
    res = model.predict(np.array([bow]))[0] # Recibe el resultado del modelo, recibiendo como entrada 'bow' (ver línea de arriba, 45)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Si 'resultado' es mayor que el umbral, entonces considera el par [indice del 'resultado', 'resultado']
    results.sort(key=lambda x: x[1], reverse=True)
    # Ordenar resultados según la mayor probabilidad (Softmax como capa de salida)
    return_list = []
    for r in results:
        return_list.append({'intent': tags[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """
    Recibe:
    - intents_list: Lista de intenciones ({intent: nombre del tag; probabilty: probabilidad según modelo})
    - intents_json: Archivo 'intents.json' cargado
    """
    tag = intents_list[0]['intent'] # Del primero (el más probable), sacar la etiqueta (nombre)
    list_of_intents = intents_json['intents'] # Lista de intenciones en el archivo json
    result = []
    for i in list_of_intents:
        # Por cada intención que existe en el archivo json
        if i['tag'] == tag:
            # Si la etiqueta es la misma a la predicha
            result = random.choice(i['responses'])
            # Elegir una respuesta
            break
        # Romper el bucle
    # Retornar respuesta de json (String)
    return result

print("Hola! Soy A.V.T... El Asistente Virtual de Turismo, dime, ¿qué puedo hacer por ti?")


class ChatBot:
    def __init__(self, msg, place_context):
        self.message = msg
        self.message = self.message.lower()
        self.ints = predict_class(self.message)
        self.place_context = place_context
        self.responses = []
        self.intencion = self.ints[0]['intent']
        self.res = "no fuciona"
        """
        ints: lista de objetos (diccionario). 
        Ejemplo: ints =
        [{'intents': nombre del tag, 'probabilty': resultado de la neurona},
        {...}]
        """
    def create_response(self):
        aux = filter(self.message, words)# Filtrar las palabras (quitar las del json) y tener lista tokenizada
        aux = ' '.join([word.lower() for word in aux])# Juntar los tokens en un solo string
        tokens = make_keywords(aux)# Filtrar de nuevo para tener solamente keywords relevantes
        tokens = [str(token) for token in tokens]# e.g.: ['Museo', 'arte', 'lima']

        #intencion = self.ints[0]['intent']
        # responses = []
        if self.intencion == "consulta_trivia":
            # print(">>> SELECT fact FROM fun_facts WHERE touristic_place_id == %s"%tokens)
            self.responses, self.place_context = fake_query(tokens, query_from="fun_facts", column_target="fact",
                                                  place_context=self.place_context)
        elif self.intencion == "consulta_lugar":
            # print(">>> SELECT province_id FROM touristic_place WHERE name == %s"%tokens)
            self.responses, self.place_context = fake_query(tokens, query_from="touristic_place", column_target="province_id",
                                                  place_context=self.place_context)
        elif self.intencion == "consulta_tiempo":
            # print(">>> SELECT schedule_info FROM touristic_place WHERE name == %s"%tokens)
            self.responses, self.place_context = fake_query(tokens, query_from="touristic_place", column_target="schedule_info",
                                                  place_context=self.place_context)
        elif self.intencion == "consulta_precio":
            # print(">>> SELECT price FROM touristic_place WHERE name == %s"%tokens)
            self.responses, self.place_context = fake_query(tokens, query_from="touristic_place", column_target="price",
                                                  place_context=self.place_context)
        elif self.intencion == "consulta_lugares_cerca":
            self.responses = query_near(get_user_location())
            self.place_context = self.responses
            print("Encontré... %s" % self.responses)


    def select_response(self):
        if len(self.responses) > 0:
            # Si se hizo una consulta que sí devuelve info:
            i = random.randint(0, len(self.responses) - 1)  # Elegir respuesta al azar
            if self.intencion == "consulta_lugar":
                print(">> No tengo GPS, pero sé que queda en %s" % self.responses[i])
            print(">> ", self.responses[i])
            self.res = self.responses[i]
        else:
            self.res = get_response(self.ints, intents)
            print(">>", self.res)
            # if ints[0]['intent'] == "despedida":