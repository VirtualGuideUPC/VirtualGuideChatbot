# Chatbot para el proyecto 'Virtual Guide UPC'. 
Usando redes neuronales, nltk y spacy.

## Instalación de las librerías de training.py:
pip install -U spacy
python -m spacy download es_core_news_sm

## Instrucciones de uso
* Si el archivo intents.json es modificado, ejecutrar preparacion.py y training.py, en ese orden.
* Utilizar solamente training.py para los cambios en la red neuronal o como referencia para crear otro modelo.
* Ejecutar chatbot.py para la demostración del chatbot, usando el modelo 'chatbot_model.model'