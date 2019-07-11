import tensorflow as tf

tf.enable_eager_execution()

x = tf.zeros((10, 10))
x += 2

print(x)
