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

from recursos import get_user_location, lemmatizer, filter, make_keywords, new_query, query_near, select_names

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


class ChatBot:
    def __init__(self, msg: str, place_context: str):
        self.message = msg.lower()
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
    
    def set_message(self, msg):
        """
        Analiza el mensaje de entrada hasta extraer una lista de palabras clave.
        >> msg: Mensaje (string) de entrada
        Retorna: Lista de keywords
        """
        self.message = msg.lower() #...... Mensaje de entrada en minúsculas
        self.ints = predict_class(self.message) #.. Intención (objeto)
        self.intencion = self.ints[0]['intent'] #.. Intención (string)
        aux = filter(self.message, words) # Filtrar las palabras para tener lista tokenizada
        aux = ' '.join([word.lower() for word in aux])
        self.responses = []
        tokens = make_keywords(aux) # Filtro (sintaxis)
        tokens = [str(token) for token in tokens] # e.g.: ['Museo', 'arte', 'lima']
        return select_names(keywords=tokens, place_context=self.place_context) # Lista de lugares candidatos
    def select_candidate(self, place_candidates):
        """
        De una lista de lugares 'candidatos', elige o pregunta al usuario
        """
        if len(place_candidates) == 1:
            self.place_context = place_candidates[0]
        else:
            print(">> Quiero asegurarme de entenderte bien, ¿a cuál de estos lugares te refieres?")
            for i in range(len(place_candidates)):
                print("%s: %s"%(i, place_candidates[i]))
            message = input("... Por favor, escribe el número: ")
            self.place_context = place_candidates[int(message)]
    def create_response(self):
        """
        Crea los valores en la lista self.responses
        (Solamente para las consultas relacionadas a la base de datos)
        """
        aux_context = "'%s'"%self.place_context
        if self.intencion == "consulta_trivia":
            res = new_query(select_column=['fact'], from_data = "fun_facts", where_pairs=[("touristic_place_id", aux_context)])
            self.responses = [trivia for trivia in res['fact']].copy()
        elif self.intencion == "consulta_lugar":
            res = new_query(select_column=['longitude', 'latitude'], from_data = "touristic_place", where_pairs=[("name", aux_context)])
            if len(res) > 0:
                self.responses = ["Las coordenadas de %s son (%s, %s)"%(self.place_context, res.values[0][0], res.values[0][1])].copy()
        elif self.intencion == "consulta_tiempo":
            res = new_query(['schedule_info'], "touristic_place", [("name", aux_context)])
            if len(res) > 0:
                self.responses = [res.values[0][0]]
        elif self.intencion == "consulta_precio":
            res = new_query(['cost_info', 'price'], "touristic_place", [("name", aux_context)])
            if len(res) > 0:
                self.responses = ["%s; con precio de %s"%(res.values[0][0], res.values[0][1])]
        elif self.intencion == "contexto":
            if self.place_context != " ":
                self.responses = [self.place_context]
        elif self.intencion == "consulta_lugares_cerca":
            res = query_near(get_user_location())
            self.place_context = res
            self.responses = ["Encontré: %s"%res]
    def select_response(self):
        if len(self.responses) > 0:
            # Si se hizo una consulta que sí devuelve info:
            self.res = random.choice(self.responses) # Elegir respuesta al azar
            return
        self.res = get_response(self.ints, intents)

    def run(self):
        """
        Ejecuta el chatbot en un bucle hasta que sea la despedida
        """
        print("Hola! Soy A.V.T... El Asistente Virtual de Turismo, dime, ¿qué puedo hacer por ti?")
        while True:
            place_candidates = self.set_message(input("")) # Retorna los posibles lugares del que usuario habla
            #print("Candidates: ", place_candidates)
            self.select_candidate(place_candidates)
            #print("Context after that: ", self.place_context)
            self.create_response()
            #print("Responses after that: ", self.responses)
            self.select_response()
            print(">>", self.res)
            if self.intencion == "despedida":
                break
        return 0

"""
print("Hola! Soy A.V.T... El Asistente Virtual de Turismo, dime, ¿qué puedo hacer por ti?")
place_context = " " # Contexto: (Para continuar consultas sobre un mismo lugar :p)
while True:
    message = input("")
    message = message.lower()
    ints = predict_class(message)
    aux = filter(message, words) # Filtrar las palabras (quitar las del json) y tener lista tokenizada
    aux = ' '.join([word.lower() for word in aux]) # Juntar los tokens en un solo string 
    tokens = make_keywords(aux) # Filtrar de nuevo para tener solamente keywords relevantes
    tokens = [str(token) for token in tokens] # e.g.: ['Museo', 'arte', 'lima']
    intencion = ints[0]['intent']
    responses = []

    if intencion == "despedida":
        # Si se está despidiendo, responde y corta el bucle
        res = get_response(ints, intents)
        print(">>", res)
        break

    # Si pregunta por el lugar más cercano, no necesita keywords
    if intencion == "consulta_lugares_cerca":
        responses = query_near(get_user_location())
        place_context = responses
        print(">> Encontré... %s"%responses)
        continue

    # USAR KEYWORDS:
    place_candidates = select_names(keywords=tokens, place_context=place_context) # Lista de lugares candidatos
    if len(place_candidates) == 1:
        place_context = place_candidates[0]
    else:
        print(">> Quiero asegurarme de entenderte bien, ¿a cuál de estos lugares te refieres?")
        for i in range(len(place_candidates)):
            print("%s: %s"%(i, place_candidates[i]))
        message = input("... Por favor, escribe el número: ")
        place_context = place_candidates[int(message)]
    aux_context = "'%s'"%place_context
    responses = [] # Por defecto, 0 respuestas

    if intencion == "consulta_trivia":
        #responses, place_context = fake_query(tokens, query_from="fun_facts", column_target="fact", place_context=place_context)
        responses = new_query(select_column=['fact'], from_data = "fun_facts", where_pairs=[("touristic_place_id", aux_context)])
        responses = [trivia for trivia in responses['fact']]
    elif intencion == "consulta_lugar":
        #responses, place_context = fake_query(tokens, query_from="touristic_place", column_target="province_id", place_context=place_context)
        responses = new_query(select_column=['longitude', 'latitude'], from_data = "touristic_place", where_pairs=[("name", aux_context)])
        if len(responses) > 0:
            responses = ["Las coordenadas de %s son (%s, %s)"%(place_context, responses.values[0][0], responses.values[0][1])]
    elif intencion == "consulta_tiempo":
        #responses, place_context = fake_query(tokens, query_from="touristic_place", column_target="schedule_info", place_context=place_context)
        responses = new_query(['schedule_info'], "touristic_place", [("name", aux_context)])
        if len(responses) > 0:
            responses = [responses.values[0][0]]
    elif intencion == "consulta_precio":
        #print(">>> SELECT price FROM touristic_place WHERE name == %s"%tokens)
        #responses, place_context = fake_query(tokens, query_from="touristic_place", column_target="price", place_context=place_context)
        responses = new_query(['cost_info', 'price'], "touristic_place", [("name", aux_context)])
        if len(responses) > 0:
            responses = ["%s; con precio de %s"%(responses.values[0][0], responses.values[0][1])]
    elif intencion == "contexto":
        if place_context != " ":
            responses = [place_context]
    #...
    if len(responses) > 0:
        # Si se hizo una consulta que sí devuelve info:
        i = random.randint(0, len(responses) - 1) # Elegir respuesta al azar
        print(">> ", responses[i])
        continue
    # Por defecto (0 respuestas)
    res = get_response(ints, intents)
    print(">>", res)
"""

AVT = ChatBot(" ", " ")
AVT.run()