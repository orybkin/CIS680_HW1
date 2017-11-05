import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def task(activation, with_CE, folder, range):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    granularity = 100

    # Computation graph
    x = tf.placeholder(tf.float32)
    w = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    output = activation(w * x + b)
    loss_l2 = (tf.square(output - y))
    grad_x, grad_w, grad_b = tf.gradients(loss_l2, [x, w, b])


    # Subtask 1
    w_value = np.array([np.array([np.linspace(-range, range, granularity)])]).transpose([1, 0, 2])
    b_value = np.array([np.array([np.linspace(-range, range, granularity)])]).transpose([1, 2, 0])
    w_value = np.repeat(w_value, granularity, axis=1)
    b_value = np.repeat(b_value, granularity, axis=2)

    # Run graph
    with tf.Session() as sess:
        feed_dict = {x: 1,
                     y: .5,
                     w: w_value,
                     b: b_value}
        tf_out = sess.run([loss_l2, output, grad_x, grad_w, grad_b], feed_dict=feed_dict)

    surf = ax.plot_surface(w_value[0], b_value[0], tf_out[1][0], cmap=cm.coolwarm)
    #ax.set_title('Sigmoid activation')
    ax.set_xlim(-range, range)
    ax.set_ylim(-range, range)
    ax.set_xlabel('weights')
    ax.set_ylabel('biases')
    ax.set_zlabel('activation')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig = plt.draw()
    plt.savefig(folder + 'sigmoid.eps')


    # Subtask 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf=ax.plot_surface(w_value[0],b_value[0],tf_out[0][0],cmap=cm.coolwarm)

    #ax.set_title('L2 loss')
    ax.set_xlim(-range,range)
    ax.set_ylim(-range,range)
    ax.set_xlabel('weights')
    ax.set_ylabel('biases')
    ax.set_zlabel('loss')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig = plt.draw()
    plt.savefig(folder + 'l2_loss.eps')

    # Subtask 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf=ax.plot_surface(w_value[0],b_value[0],tf_out[3][0],cmap=cm.coolwarm)

    ##ax.set_title('Gradient of L2 loss w.r.t. weight')
    ax.set_xlim(-range,range)
    ax.set_ylim(-range,range)
    ax.set_xlabel('weights')
    ax.set_ylabel('biases')
    ax.set_zlabel('gradient')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig = plt.draw()
    plt.savefig(folder + 'l2_gradient.eps')


    if with_CE:
        # Subtask 4
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        loss_CE = -(y * tf.log(output) + (1 - y) * tf.log(1 - output))
        grad_x, grad_w, grad_b = tf.gradients(loss_CE, [x, w, b])

        # Run graph
        with tf.Session() as sess:
            feed_dict = {x: 1,
                         y: .5,
                         w: w_value,
                         b: b_value}
            tf_out = sess.run([loss_CE, output, grad_x, grad_w, grad_b], feed_dict=feed_dict)

        surf = ax.plot_surface(w_value[0], b_value[0], tf_out[0][0], cmap=cm.coolwarm)
        ##ax.set_title('Cross entropy loss')
        ax.set_xlim(-range, range)
        ax.set_ylim(-range, range)
        ax.set_xlabel('weights')
        ax.set_ylabel('biases')
        ax.set_zlabel('loss')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig = plt.draw()
        plt.savefig(folder + 'CE_loss.eps')


        # Subtask 5
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf=ax.plot_surface(w_value[0],b_value[0],tf_out[3][0],cmap=cm.coolwarm)

        ##ax.set_title('Gradient of cross entropy w.r.t. weight')
        ax.set_xlim(-range,range)
        ax.set_ylim(-range,range)
        ax.set_xlabel('weights')
        ax.set_ylabel('biases')
        ax.set_zlabel('gradient')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig = plt.draw()
        plt.savefig(folder + 'CE_gradient.eps')


def my_leaky_relu(x):
    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)

task(tf.sigmoid, True, './task1/', 6)
task(my_leaky_relu, False, './task4/', 1)



