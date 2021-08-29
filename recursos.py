## Librerías:
# Procesamiento de Lenguaje Natural
import spacy # Lemmatizer (convertir palabras) con lenguaje español
import numpy as np
import nltk # Natural Language ToolKit: Tokenizar

# Preparación
nltk.download('punkt')

# Lematizador que acepta texto con Spacy español
nlp = spacy.load('es_core_news_sm')
def lemmatizer(text):
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])