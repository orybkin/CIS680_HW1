import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def task(activation, folder):
    tf.reset_default_graph()
    batch_size = 64
    reuse = False

    # Subtask 1
    with tf.variable_scope('input', reuse=reuse):
        input_x = tf.placeholder(tf.float32, [batch_size, 16, 16, 1])
        input_y = tf.placeholder(tf.int32, [batch_size, 1])

    print(input_x)
    with tf.variable_scope('conv1', reuse=reuse):
        W = tf.get_variable('weights', [7, 7, 1, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 1, 1, 16], initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(input_x, W, [1, 1, 1, 1], 'VALID') + b
        x = activation(x)

    print(x)
    with tf.variable_scope('conv2', reuse=reuse):
        W = tf.get_variable('weights', [7, 7, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 1, 1, 8], initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') + b
        x = activation(x)

    print(x)
    # first branch
    with tf.variable_scope('fc1', reuse=reuse):
        W = tf.get_variable('weights', [128, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
        x_classif = tf.matmul(tf.reshape(x, [batch_size, -1]), W) + b
    print(x_classif)

    with tf.variable_scope('sm1', reuse=reuse):
        z = tf.cast(input_y, dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_classif, labels=tf.cast(input_y, dtype=tf.float32)))
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round(tf.sigmoid(x_classif)), dtype=tf.int32), input_y)))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # variables
    x_train = np.array([np.load('./datasets/line/line_imgs.npy').astype(np.float32)]).transpose([1, 2, 3, 0])
    y_train = np.array([np.load('./datasets/line/line_labs.npy')]).T.astype(np.int32)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    pro_loss = np.zeros(10000)
    pro_accuracy = np.zeros(10000)
    for i in np.arange(1000):
        dummy, curr_loss, pro_accuracy[i] = sess.run([train, loss, accuracy], {input_x: x_train, input_y: y_train})
        pro_loss[i] = curr_loss.mean()
        if pro_accuracy[i] == 1: break

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(pro_loss[0:i])
    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    #f.suptitle('One-task convolutional network')
    ax2.plot(pro_accuracy[0:i + 1])
    plt.draw()
    f.set_size_inches(f.get_size_inches()[0],f.get_size_inches()[1]/2)
    plt.tight_layout()
    plt.savefig(folder + 'one_task.eps')




    tf.reset_default_graph()
    batch_size = 64
    reuse = False

    # Subtask 2
    with tf.variable_scope('input', reuse=reuse):
        input_x = tf.placeholder(tf.float32, [batch_size, 16, 16, 1])
        input_y = tf.placeholder(tf.int32, [batch_size, 1])
        input_width = tf.placeholder(tf.float32, [batch_size, 1])

    print(input_x)
    with tf.variable_scope('conv1', reuse=reuse):
        W = tf.get_variable('weights', [7, 7, 1, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 10, 10, 16], initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(input_x, W, [1, 1, 1, 1], 'VALID') + b
        x = activation(x)

    print(x)
    with tf.variable_scope('conv2', reuse=reuse):
        W = tf.get_variable('weights', [7, 7, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 4, 4, 8], initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') + b
        x = activation(x)

    print(x)
    # first branch
    with tf.variable_scope('fc1', reuse=reuse):
        W = tf.get_variable('weights', [128, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
        x_classif = tf.matmul(tf.reshape(x, [batch_size, -1]), W) + b
    print(x_classif)

    with tf.variable_scope('cl', reuse=reuse):
        z = tf.cast(input_y, dtype=tf.float32)
        loss_cl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_classif, labels=tf.cast(input_y, dtype=tf.float32)))
        accuracy_cl = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.round(tf.sigmoid(x_classif)), dtype=tf.int32),
                                                          input_y)))

    # Second branch
    with tf.variable_scope('fc2', reuse=reuse):
        W = tf.get_variable('weights', [128, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer(0.1))
        x_regress = tf.matmul(tf.reshape(x, [batch_size, -1]), W) + b
    print(x_regress)

    with tf.variable_scope('re', reuse=reuse):
        loss_re = tf.reduce_mean(tf.nn.l2_loss(x_regress - input_width))
        accuracy_re = tf.reduce_mean(tf.to_float(tf.abs(x_regress - input_width) < 0.5))

    total_loss = loss_cl + 0.01 * loss_re

    # IO
    tf.summary.scalar('mean', tf.reduce_mean(total_loss))
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(total_loss)

    # variables
    x_train = np.array([np.load('./datasets/detection/detection_imgs.npy').astype(np.float32)]).transpose([1, 2, 3, 0])
    y_train = np.array([np.load('./datasets/detection/detection_labs.npy')]).T.astype(np.int32)
    width_train = np.array([np.load('./datasets/detection/detection_width.npy')]).T

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    pro_losscl = np.zeros(10000)
    pro_accuracycl = np.zeros(10000)
    pro_lossre = np.zeros(10000)
    pro_accuracyre = np.zeros(10000)
    for i in np.arange(10000):
        summary, dummy, curr_losscl, pro_accuracycl[i], curr_lossre, pro_accuracyre[i] = sess.run(
            [merged, train, loss_cl, accuracy_cl, loss_re, accuracy_re],
            {input_x: x_train, input_y: y_train, input_width: width_train})
        pro_losscl[i] = curr_losscl.mean()
        pro_lossre[i] = curr_lossre.mean()

        train_writer.add_summary(summary, i)
        if pro_accuracycl[i] == 1 and pro_accuracyre[i] == 1: break

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(pro_losscl[0:i])
    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')
    ax1.set_ylabel('CE loss')
    ax2.set_ylabel('accuracy')
    ax2.plot(pro_accuracycl[0:i + 1])
    ax3.plot(pro_lossre[0:i])
    ax3.set_xlabel('epoch')
    ax4.set_xlabel('epoch')
    ax3.set_ylabel('L2 loss')
    ax4.set_ylabel('accuracy')
    ax4.plot(pro_accuracyre[0:i + 1])
    #f.suptitle('Two-tasks convolutional network')
    plt.draw()
    plt.tight_layout()
    plt.savefig(folder + 'two_tasks.eps')


def my_leaky_relu(x):
    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)

task(tf.nn.relu, './task3/')
task(my_leaky_relu, './task4/')

