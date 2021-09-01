# 3. Red Neuronal
## Librerías:

import pickle # Guardar archivos
import numpy as np

# Redes Neuronales:
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Cargar Data de Entrenamiento
training = pickle.load(open('training.pkl','rb'))

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