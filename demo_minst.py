import tensorflow as tf
from tensorflow import keras
import os

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


model = create_model()
path="training_2/cp-{epoch:04d}.ckpt"
dir=os.path.dirname(path)
cp_callback=keras.callbacks.ModelCheckpoint(path,save_weights_only=True,verbose=1,period=5)
model.fit(train_images,train_labels,validation_data=(test_images,test_labels),callbacks=[cp_callback],epochs=50)

