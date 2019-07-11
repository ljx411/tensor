import tensorflow as tf

tf.enable_eager_execution()
x = tf.ones((2, 2))
with tf.GradientTape() as g:
    g.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
dz_dy = g.gradient(z, x)
# assert dz_dy.numpy() == 8.0
print(dz_dy)