import tensorflow as tf
from custom_datasets import MNISTConditionedLabel, MNISTConditionedPrevNum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


dataset = MNISTConditionedLabel()
# dataset = MNISTConditionedPrevNum()
WIDTH = 28
HEIGHT = 28
mb_size = 128
Z_dim = 32
X_dim = dataset.train.images.shape[1]
y_dim = dataset.train.labels.shape[1]
h_dim = 128


""" Helper functions"""
def conv_transpose(img, num_outputs, out_width, out_height, kernel_size, stride, padding="SAME", activation_fn=tf.nn.relu, scope=None):
    with tf.variable_scope(scope):
        output_shape = (tf.shape(img)[0], out_height, out_width, num_outputs)
        strides = (1, stride, stride, 1)
        kernel_shape = tuple(kernel_size) + (num_outputs, img.get_shape()[-1])
        kernel = tf.get_variable(name='kernel',
                              shape=kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                              )

        bias = tf.get_variable(name='bias',
                            shape=(num_outputs,),
                            initializer=tf.constant_initializer(0.0),
                            )
        conv_out = tf.nn.conv2d_transpose(img, kernel, output_shape=output_shape, strides=strides, padding=padding)
        conv_out = tf.reshape(conv_out, [-1, out_height, out_width, num_outputs])
        conv_out = tf.nn.bias_add(conv_out, bias)
        if activation_fn is not None:
            conv_out = activation_fn(conv_out)
        return conv_out


def lrelu(x, leak=0.2, name="lrelu"):
    # leaky RELU unit
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


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


""" Encoder model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

def recognition(x, y):
    ### MNIST conditioned on label v1
    inputs = tf.concat(concat_dim=1, values=[x, y])
    with tf.variable_scope("recognition"):
        R_h1 = tf.contrib.slim.fully_connected(inputs, h_dim, activation_fn=tf.nn.relu)

        mu = tf.contrib.slim.fully_connected(R_h1, Z_dim, activation_fn=None, scope="mu")
        log_sigma = tf.contrib.slim.fully_connected(R_h1, Z_dim, activation_fn=None, scope="log_sigma")

    return mu, log_sigma


""" Prior model """
def prior(y):
    ### MNIST conditioned on label v1
    inputs = y
    with tf.variable_scope("prior"):
        R_h1 = tf.contrib.slim.fully_connected(inputs, h_dim, activation_fn=tf.nn.relu)

        mu = tf.contrib.slim.fully_connected(R_h1, Z_dim, activation_fn=None, scope="mu")
        log_sigma = tf.contrib.slim.fully_connected(R_h1, Z_dim, activation_fn=None, scope="log_sigma")

    return mu, log_sigma


""" Decoder model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

def generation(z, reuse=False):
    ### MNIST conditioned on label v1
    inputs = z
    with tf.variable_scope("generation", reuse=reuse):
        G_h1 = tf.contrib.slim.fully_connected(inputs, h_dim, activation_fn=tf.nn.relu)
        G_logit = tf.contrib.slim.fully_connected(G_h1, X_dim, activation_fn=None)
    G_prob = tf.nn.sigmoid(G_logit)
    return G_prob, G_logit

def sample_Z(m, n):
    return np.random.normal(0, 1, size=[m, n])


""" Complete model """
mu_prior, log_sigma_prior = prior(y)
z_prior = mu_prior + (tf.exp(log_sigma_prior/2.) * Z)
G_output_prior, G_logit_prior = generation(z_prior)

mu, log_sigma = recognition(X, y)
z = mu + (tf.exp(log_sigma/2.) * Z)
G_output, G_logit = generation(z, reuse=True)


""" Loss """
generation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=X, logits=G_logit), 1)
latent_loss = 0.5 * tf.reduce_sum(tf.square(mu_prior-mu)/tf.exp(log_sigma_prior) + tf.exp(log_sigma-log_sigma_prior) + log_sigma_prior - log_sigma - 1, 1)
# For reference, here's the non-conditional loss:
#     latent_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_sigma) - log_sigma - 1, 1)
V_loss = tf.reduce_mean(generation_loss + latent_loss)
V_solver = tf.train.AdamOptimizer(0.001).minimize(V_loss)


""" Train """
sess = tf.Session()
sess.run(tf.initialize_all_variables())

if not os.path.exists('out/'):
    os.makedirs('out/')


X_vis, y_vis = dataset.train.next_batch(16)
fig = plot(X_vis)
plt.savefig("out/base.png", bbox_inches='tight')
plt.close(fig)

for epoch in range(100):
    for idx in range(int(dataset.train.num_examples/mb_size)):
        X_mb, y_mb = dataset.train.next_batch(mb_size)
        Z_sample = sample_Z(mb_size, Z_dim)
        _, G_loss_curr, L_loss_curr = sess.run((V_solver, generation_loss, latent_loss), feed_dict={X: X_mb, Z:Z_sample, y: y_mb})

    print ("Epoch: {}; G_loss: {:.4}; L_loss: {:.4}".format(epoch, np.mean(G_loss_curr), np.mean(L_loss_curr)))

    Z_sample = sample_Z(16, Z_dim)
    samples = sess.run(G_output_prior, feed_dict={Z: Z_sample, y:y_vis})
    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
