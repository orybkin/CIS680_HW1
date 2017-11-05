import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


folder = './task2/'


tf.reset_default_graph()
batch_size = 64
reuse = False

# Subtask 1
with tf.variable_scope('input', reuse=reuse):
    input_x = tf.placeholder(tf.float32, [batch_size, 4, 4])
    input_y = tf.placeholder(tf.int32, [batch_size, 1])

with tf.variable_scope('fc1', reuse=reuse):
    W = tf.get_variable('weights', [4 * 4, 4], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [batch_size, 4], initializer=tf.constant_initializer(0.1))

    x = tf.matmul(tf.reshape(input_x, [batch_size, 16]), W) + b
    x = tf.sigmoid(x)

with tf.variable_scope('fc2', reuse=reuse):
    W = tf.get_variable('weights', [4, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [batch_size, 1], initializer=tf.constant_initializer(0.1))
    x = tf.matmul(x, W) + b
    x = tf.sigmoid(x)

# Loss
with tf.variable_scope('sm', reuse=reuse):
    loss = tf.reduce_mean(tf.pow(x - tf.cast(input_y, dtype=tf.float32), 2))
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round(x), dtype=tf.int32), input_y)))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# variables
x_train = np.load('./datasets/random/random_imgs.npy').astype(np.float32)
y_train = np.array([np.load('./datasets/random/random_labs.npy')]).T.astype(np.int32)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
pro_loss = np.zeros(10000)
pro_accuracy = np.zeros(10000)
for i in np.arange(10000):
    dummy, curr_loss, pro_accuracy[i] = sess.run([train, loss, accuracy], {input_x: x_train, input_y: y_train})
    pro_loss[i] = curr_loss.mean()
    if pro_accuracy[i] == 1: break

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(pro_loss[0:i])
ax1.set_xlabel('epoch')
ax2.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
#f.suptitle('Sigmoid + L2 loss FC network')
ax2.plot(pro_accuracy[0:i + 1])
f.set_size_inches(f.get_size_inches()[0],f.get_size_inches()[1]/2)
plt.tight_layout()
plt.draw()
plt.savefig(folder + 'sigm_l2.eps')






tf.reset_default_graph()
batch_size = 64
reuse = False

# Subtask 2
with tf.variable_scope('input', reuse=reuse):
    input_x = tf.placeholder(tf.float32, [batch_size, 4, 4])
    input_y = tf.placeholder(tf.int32, [batch_size, 1])

with tf.variable_scope('fc1', reuse=reuse):
    W = tf.get_variable('weights', [4 * 4, 4], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 4], initializer=tf.constant_initializer(0.1))

    x = tf.matmul(tf.reshape(input_x, [batch_size, 16]), W) + b
    x = tf.sigmoid(x)

with tf.variable_scope('fc2', reuse=reuse):
    W = tf.get_variable('weights', [4, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
    x = tf.matmul(x, W) + b

# Loss
with tf.variable_scope('sm', reuse=reuse):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.cast(input_y, dtype=tf.float32)))
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round(tf.sigmoid(x)), dtype=tf.int32), input_y)))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# variables
x_train = np.load('./datasets/random/random_imgs.npy').astype(np.float32)
y_train = np.array([np.load('./datasets/random/random_labs.npy')]).T.astype(np.int32)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
pro_loss = np.zeros(10000)
pro_accuracy = np.zeros(10000)
for i in np.arange(10000):
    dummy, curr_loss, pro_accuracy[i] = sess.run([train, loss, accuracy], {input_x: x_train, input_y: y_train})
    pro_loss[i] = curr_loss.mean()
    if pro_accuracy[i] == 1: break

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
print(f.get_size_inches())
ax1.plot(pro_loss[0:i])
ax1.set_xlabel('epoch')
ax2.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
#f.suptitle('Sigmoid + cross-entropy loss FC network')
ax2.plot(pro_accuracy[0:i + 1])
plt.draw()
f.set_size_inches(f.get_size_inches()[0],f.get_size_inches()[1]/2)
plt.tight_layout()
plt.savefig(folder + 'sigm_ce.eps')




tf.reset_default_graph()
batch_size = 64
reuse = False

# Subtask 3
with tf.variable_scope('input', reuse=reuse):
    input_x = tf.placeholder(tf.float32, [batch_size, 4, 4])
    input_y = tf.placeholder(tf.int32, [batch_size, 1])

with tf.variable_scope('fc1', reuse=reuse):
    W = tf.get_variable('weights', [4 * 4, 4], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 4], initializer=tf.constant_initializer(0.1))

    x = tf.matmul(tf.reshape(input_x, [batch_size, 16]), W) + b
    x = tf.nn.relu(x)

with tf.variable_scope('fc2', reuse=reuse):
    W = tf.get_variable('weights', [4, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
    x = tf.matmul(x, W) + b
    x = tf.sigmoid(x)

# Loss
with tf.variable_scope('sm', reuse=reuse):
    loss = tf.reduce_mean(tf.pow(x - tf.cast(input_y, dtype=tf.float32), 2))
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round((x)), dtype=tf.int32), input_y)))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# variables
x_train = np.load('./datasets/random/random_imgs.npy').astype(np.float32)
y_train = np.array([np.load('./datasets/random/random_labs.npy')]).T.astype(np.int32)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
pro_loss = np.zeros(10000)
pro_accuracy = np.zeros(10000)
for i in np.arange(10000):
    dummy, curr_loss, pro_accuracy[i] = sess.run([train, loss, accuracy], {input_x: x_train, input_y: y_train})
    pro_loss[i] = curr_loss.mean()
    if pro_accuracy[i] == 1: break

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(pro_loss[0:i])
ax1.set_xlabel('epoch')
ax2.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
#f.suptitle('ReLU + L2 loss FC network')
ax2.plot(pro_accuracy[0:i + 1])
plt.draw()
f.set_size_inches(f.get_size_inches()[0],f.get_size_inches()[1]/2)
plt.tight_layout()
plt.savefig(folder + 'relu_l2.eps')




tf.reset_default_graph()
batch_size = 64
reuse = False

# Subtask 4
with tf.variable_scope('input', reuse=reuse):
    input_x = tf.placeholder(tf.float32, [batch_size, 4, 4])
    input_y = tf.placeholder(tf.int32, [batch_size, 1])

with tf.variable_scope('fc1', reuse=reuse):
    W = tf.get_variable('weights', [4 * 4, 4], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 4], initializer=tf.constant_initializer(0.1))

    x = tf.matmul(tf.reshape(input_x, [batch_size, 16]), W) + b
    x = tf.nn.relu(x)

with tf.variable_scope('fc2', reuse=reuse):
    W = tf.get_variable('weights', [4, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
    x = tf.matmul(x, W) + b

# Loss
with tf.variable_scope('sm', reuse=reuse):
    z = tf.cast(input_y, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x, labels=tf.cast(input_y, dtype=tf.float32)))
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round(tf.sigmoid(x)), dtype=tf.int32), input_y)))





# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# variables
x_train = np.load('./datasets/random/random_imgs.npy').astype(np.float32)
y_train = np.array([np.load('./datasets/random/random_labs.npy')]).T.astype(np.int32)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
pro_loss = np.zeros(10000)
pro_accuracy = np.zeros(10000)
for i in np.arange(10000):
    dummy, curr_loss, pro_accuracy[i] = sess.run([train, loss, accuracy], {input_x: x_train, input_y: y_train})
    pro_loss[i] = curr_loss.mean()
    if pro_accuracy[i] == 1: break

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(pro_loss[0:i])
ax1.set_xlabel('epoch')
ax2.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
#f.suptitle('ReLU + cross-entropy loss FC network')
plt.xlabel('epoch')

ax2.plot(pro_accuracy[0:i + 1])
plt.draw()
f.set_size_inches(f.get_size_inches()[0],f.get_size_inches()[1]/2)
plt.tight_layout()

plt.savefig(folder + 'relu_ce.eps')
plt.show()
