import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_minist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_minist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(256, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_accu = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)


# print('精确度为：',test_accu)
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predictions_label = np.argmax(predictions_array)
    if predictions_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predictions_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predictions_label = np.argmax(predictions_array)
    thisplot[predictions_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols * 2, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
