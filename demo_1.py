import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_minist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_minist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()