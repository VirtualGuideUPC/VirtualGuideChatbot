## Librerías:
# Procesamiento de Lenguaje Natural
from re import A
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import numpy as np
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
    doc = nlp(text)
    sentence = next(doc.sents) 
    for word in sentence:
        print("%s:%s" % (word,word.dep_))

def filter(text, list_words):
    text = lemmatizer(text)
    tokens = nltk.word_tokenize(text)
    bag = []
    for word in tokens:
        bag if word.lower() in list_words else bag.append(word)
    return bag

"""
print("GO.")
while True:
    sent = input("")
    texto = filter(sent, words)
    print(texto)
    if sent == "SALIR":
        break
    texto = ' '.join([word for word in texto])
    print(texto)
    analyze(texto)

#doc = nlp(text)
#sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]
#print(sub_toks) 
"""

print(words)