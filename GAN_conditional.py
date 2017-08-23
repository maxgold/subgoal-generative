import tensorflow as tf
from custom_datasets import MNISTConditionedLabel, MNISTConditionedPrevNum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

out_dir = 'gan_label_out'

if out_dir == 'gan_label_out':
    dataset = MNISTConditionedLabel()
elif out_dir == 'gan_full_out':
    dataset = MNISTConditionedPrevNum()


WIDTH = 28
HEIGHT = 28
mb_size = 128
Z_dim = 32
X_dim = dataset.train.images.shape[1]
y_dim = dataset.train.labels.shape[1]
h_dim = 128


""" Helper functions"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(HEIGHT, WIDTH), cmap='Greys_r')

    return fig


""" Discriminator model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])


def discriminator(x, y, reuse=False):
    ### MNIST conditioned on label v1
    inputs = tf.concat(axis=1, values=[x, y])
    with tf.variable_scope("discriminator", reuse=reuse):
        D_h1 = tf.contrib.slim.fully_connected(inputs, h_dim, activation_fn=tf.nn.relu)
        D_logit = tf.contrib.slim.fully_connected(D_h1, 1, activation_fn=None)
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])


def generator(z, y):
    ### MNIST conditioned on label v1
    inputs = tf.concat(axis=1, values=[z, y])
    with tf.variable_scope("generator"):
        G_h1 = tf.contrib.slim.fully_connected(inputs, h_dim, activation_fn=tf.nn.relu)
        # G_h2 = tf.contrib.slim.fully_connected(G_h1, h_dim, activation_fn=tf.nn.relu)
        G_log_prob = tf.contrib.slim.fully_connected(G_h1, X_dim, activation_fn=None)
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


""" Complete model """
G_sample = generator(Z, y)

D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y, reuse=True)


""" Loss """
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake))
D_loss = tf.reduce_mean(D_loss_real + D_loss_fake)
# G_loss = -tf.reduce_mean(tf.log(D_fake))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(1e-3).minimize(D_loss, var_list=tf.contrib.slim.get_variables(scope="discriminator"))
G_solver = tf.train.AdamOptimizer(1e-3).minimize(G_loss, var_list=tf.contrib.slim.get_variables(scope="generator"))


""" Train """
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# y_vis = np.zeros(shape=[16, y_dim])
# y_vis[np.arange(16), np.arange(16)%10] = 1
X_vis, y_vis = dataset.train.next_batch(16)
fig = plot(X_vis)
plt.savefig(out_dir + "/base.png", bbox_inches='tight')
plt.close(fig)

if out_dir == 'gan_full_out':
    fig = plot(y_vis)
    plt.savefig(out_dir + '/base_condition.png', bbox_inches='tight')
    plt.close(fig)


for epoch in range(500):
    G_loss_avg = 0
    D_loss_avg = 0
    num_batches = int(dataset.train.num_examples/mb_size)
    for idx in range(num_batches):
        X_mb, y_mb = dataset.train.next_batch(mb_size)
        # Train discriminator to convergence (in practice, usually 1 iteration is used)
        for _ in range(1):
            Z_sample = sample_Z(mb_size, Z_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
        D_loss_avg += D_loss_curr/num_batches
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})
        # Train generator
        for _ in range(1):
            #print(_)
            Z_sample = sample_Z(mb_size, Z_dim)
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})
        G_loss_avg += G_loss_curr/num_batches

    print ("Epoch: {}; D_loss: {:.4}; G_loss: {:.4}".format(epoch, D_loss_avg, G_loss_avg))

    Z_sample = sample_Z(16, Z_dim)
    samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_vis})
    fig = plot(samples)
    plt.savefig(out_dir + '/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)















