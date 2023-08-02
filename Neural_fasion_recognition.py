"""
Ta siec neuronowa rozpoznaje ikony ubrań w postaci 28px x 28px
input_layer = 28x28 = 784
output_layer = 10 (9 rodzajów ubrań)
HIDDEN_LAYERS = 128 (zwiekszamy complexy naszej sieci neuronowej, sami se to wybieramy najczesciej jako procent inputu(25%-50%)

activation-
"""



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top','Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

#Tworzenie sieci
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #input- flatten-po prostu z 2 wymiarowej tablicy robi 1 wymiarow
    keras.layers.Dense(128, activation="relu"), #hidden - actiwation to po prostu funkcja zwikszajaca zlozonosc algorytmu
    keras.layers.Dense(10, activation="softmax") #output - softmax powoduje ze wartosci 10 neuronow dodaja sie do 1
])
#Włączanie modelu
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #metrics- na co zwracamy uwage

#Training model
model.fit(train_images, train_labels, epochs=5) #epoch - ile razy zobaczy ten sam obraz(bo fit bierze losowo dane) zwieksza efektywanosc algorytmu

#Testowanie modelu
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc: ", test_acc) #epoch=50 acc-0.8844 epoch=5 acc-0.8716


prediction = model.predict(test_images) #showing output layer

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
