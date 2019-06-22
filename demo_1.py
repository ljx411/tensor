import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_minist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_minist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()