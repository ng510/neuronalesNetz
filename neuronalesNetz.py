# Einfach Neuronales Netz, welches auf den FashionMNIST Daten trainiert wird
# Vorhersage, ob das Bild ein Tshirt/Top ist, oder nicht
# FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

import gzip
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)
    
X_train = open_images("../data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("../data/fashion/train-labels-idx1-ubyte.gz")

y_train = y_train == 0 #Setze alle labels auf True, falls Tshirt, sonst false

X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_labels("../data/fashion/t10k-labels-idx1-ubyte.gz")

y_test = y_test == 0


#Ein Hidden Layer mit 100 Neuronen, danach sofort ein Neuron als Ausgang.
#Input von 28*28 = 784 (da ein Bild 28*28 Pixel)
model = Sequential()

model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

#model fitten mittels 15 Durchläufen (Epochs)
#Gewichte anpassen nach 5000 Datensätzen (batch)
model.fit(
    X_train.reshape(60000, 784),
    y_train,
    epochs=15,
    batch_size=5000)

#Evaluiere Genauigkeit des Models
#Ausgabe: ['loss', 'acc'] => Genauigkeit in Prozent im 2. Eintrag 
print(model.evaluate(X_test.reshape(-1, 784), y_test))

#Grafische Ausgabe und manuelles Testen
plt.imshow(X_train[30], cmap="gray_r")
plt.show()

#Wahrscheinlichkeit, dass es sich um ein Thirt handelt
print(model.predict(X_train[30].reshape(1, 784)))