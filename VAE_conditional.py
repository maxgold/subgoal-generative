import tensorflow as tf
from custom_datasets import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import pickle

#out_dir = 'vae_label_out'

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
        #plt.scatter([6,8], [5,7], color='red', s=40)

    return fig

def plot_pretty(samples, samples_y):
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
        cur_y = samples_y[i,:].reshape(HEIGHT, WIDTH)
        obstacles = np.where(cur_y == 1)
        goal      = np.where(cur_y == .5)
        cur_pos   = np.where(cur_y == .25)
        for x, y in zip(obstacles[0],obstacles[1]):
            plt.scatter(y, x, color='red', s=30)
        for x, y in zip(goal[0], goal[1]):
            plt.scatter(y, x, color='cyan', s=30)
        for x, y in zip(cur_pos[0], cur_pos[1]):
            plt.scatter(y, x, color='green', s=30)

    return fig

def split_train_test(robot, goal, obj, split = .9):
    keys = list(robot.keys())
    train_size = int(split*len(keys))
    train_rob = {}
    test_rob  = {}
    train_goal = {}
    test_goal  = {}
    train_obj = {}
    test_obj  = {}

    random.shuffle(keys)
    for i in keys[:train_size]:
        train_rob[i] = robot[i]
        train_goal[i]   = goal[i]
        train_obj[i]    = obj[i]
    for j in keys[train_size:]:
        test_rob[j] = robot[j]
        test_goal[j] = goal[j]
        test_obj[j] = obj[j]
    return(train_rob,train_goal,train_obj,test_rob,test_goal,test_obj)



anneal = False
out_dir = 'grid_out_l1'

if out_dir == 'vae_label_out':
    dataset = MNISTConditionedLabel()
elif out_dir == 'vae_full_out':
    dataset = MNISTConditionedPrevNum()

# robot_paths = pickle.load(open('./grid_world_data/rob_paths_1000.p', 'rb'))
# goal_grids  = pickle.load(open('./grid_world_data/goal_grids_1000.p', 'rb'))
# obj_grids   = pickle.load(open('./grid_world_data/obj_grids_1000.p', 'rb'))

# train_rob,train_goal,train_obj,test_rob,test_goal,test_obj = split_train_test(robot_paths, goal_grids, obj_grids, .9)

data = load_dataset('medium9_2')
Xd_train, yd_train, Xd_test, yd_test = parse_sokoban_train_test(data,5000,1000)

# My Xd and yd are reversed...
# Xd_train, yd_train = parse_grid_data(train_rob, train_goal, train_obj, 10)
# Xd_test, yd_test = parse_grid_data_test(test_rob, test_goal, test_obj, 1)

#Xd_train, yd_train = parse_grid_data_fulltraj(train_rob, train_goal, train_obj)
#Xd_test, yd_test = parse_grid_data_fulltraj(test_rob, test_goal, test_obj)

train = DataSet_grid(Xd_train, yd_train)
test = DataSet_grid(Xd_test, yd_test)
dataset = DataSet_wrapper(train, test)



WIDTH   = 9
HEIGHT  = 9
mb_size = 128
Z_dim   = 32
X_dim   = dataset.train.x_dim
y_dim   = dataset.train.y_dim
h_dim   = 128





""" Encoder model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

def recognition(x, y):
    ### MNIST conditioned on label v1
    inputs = tf.concat(axis=1, values=[x, y])
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
G_softmax = tf.nn.softmax(G_logit)

""" Loss """
#generation_loss = 1tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=G_logit), 1)
generation_loss  = 1*tf.reduce_sum(tf.abs(X - G_softmax), axis=1)
latent_loss = 0 * tf.reduce_sum(tf.square(mu_prior-mu)/tf.exp(log_sigma_prior) + tf.exp(log_sigma-log_sigma_prior) + log_sigma_prior - log_sigma - 1, 1)
# For reference, here's the non-conditional loss:
#     latent_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_sigma) - log_sigma - 1, 1)

latent_weight = tf.placeholder(tf.float32)
V_loss  = tf.reduce_mean(generation_loss + latent_weight*latent_loss)
V_solver = tf.train.AdamOptimizer(1e-3).minimize(V_loss)


""" Train """
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)



X_vis, y_vis = dataset.test.next_batch(16)
y_vis1 = y_vis[:,:81]
y_vis2 = y_vis[:,81:162]
y_vis3 = y_vis[:,162:243]
y_vis_new = .25*y_vis1 + .5*y_vis2 + y_vis3


fig = plot(X_vis)
plt.savefig(out_dir + '/base.png', bbox_inches='tight')
plt.close(fig)


if anneal:
    latent_coefs = np.flip(np.arange(250,500,1)/500,axis=0)
    latent_coefs = np.r_[np.ones(250), latent_coefs]
else:
    latent_coefs = np.ones(500)
coef_ind = 0


if out_dir != 'vae_label_out':
    fig = plot_pretty(X_vis, y_vis_new)
    plt.savefig(out_dir + '/base_condition.png', bbox_inches='tight')
    plt.close(fig)

for epoch in range(100):
    latent_coef = latent_coefs[epoch]

    for idx in range(int(dataset.train._num_examples/mb_size)):
        X_mb, y_mb = dataset.train.next_batch(mb_size)

        Z_sample = sample_Z(X_mb.shape[0], Z_dim)
        

        _, G_loss_curr, L_loss_curr = sess.run((V_solver, generation_loss, latent_loss), feed_dict={X: X_mb, Z:Z_sample, y: y_mb, latent_weight: latent_coef})

    print ("Epoch: {}; G_loss: {:.4}; L_loss: {:.4}".format(epoch, np.mean(G_loss_curr), np.mean(L_loss_curr)))

    Z_sample = sample_Z(16, Z_dim)
    samples = sess.run(G_output_prior, feed_dict={Z: Z_sample, y:y_vis})
    fig = plot_pretty(samples, y_vis_new)
    plt.savefig(out_dir + '/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)



def refresh_image(i, dist = 1):
    X_vis, y_vis = dataset.test.next_batch(16)
    y_vis1 = y_vis[:,:81]
    y_vis2 = y_vis[:,81:162]
    y_vis3 = y_vis[:,162:243]
    y_vis_new = .25*y_vis1 + .5*y_vis2 + y_vis3


    samples = np.zeros([X_vis.shape[0],81])
    for _ in range(dist):
        Z_sample = sample_Z(X_vis.shape[0], Z_dim)
        samples += sess.run(G_output_prior, feed_dict={Z: Z_sample, y:y_vis})
    samples_m = samples/dist
    fig = plot_pretty(samples_m, y_vis_new)
    plt.savefig(out_dir +  '/z' + str(i) + 'generated.png', bbox_inches='tight')
    plt.close(fig)



for i in range(20):
    refresh_image(i, dist=50)
    print(i)






