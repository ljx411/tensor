import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


# keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post')


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

baseline_history = baseline_model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=2,
                                      validation_data=(test_data, test_labels))
# smaller_model = keras.Sequential([
#     keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(4, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
#
# smaller_history = smaller_model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=2,
#                                     validation_data=(test_data, test_labels))
# bigger_model = keras.Sequential([
#     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
#
# bigger_history = bigger_model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=2,
#                                   validation_data=(test_data, test_labels))


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + ' Train')
    plt.xlabel='Epoches'
    plt.ylabel=key.replace('_','').title()
    plt.legend()
    plt.xlim([0,max(history.epoch)])

# plot_history([
#     ('baseline',baseline_history),
#     ('smaller',smaller_history),
#     ('bigger',bigger_history)
# ])
# _12_model = keras.Sequential([
#     keras.layers.Dense(512,kernel_regularizer=keras.regularizers.l2(0.001) ,activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# _12_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
#
# _12_history = _12_model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=2,
#                                validation_data=(test_data, test_labels))
drop_model = keras.Sequential([
    keras.layers.Dense(512 ,activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation=tf.nn.relu,),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
drop_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

drop_history = drop_model.fit(train_data, train_labels, batch_size=512, epochs=20, verbose=2,
                             validation_data=(test_data, test_labels))