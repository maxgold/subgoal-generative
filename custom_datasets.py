from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import random


class DataSet_wrapper(object):
    def __init__(self, train, test):
        self.train = train
        self.test  = test


def MNISTConditionedLabel():
    return input_data.read_data_sets('../../MNIST_data', one_hot=True)


class MNISTConditionedPrevNum(object):
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        label2images = {}
        for i in range(10):
            idx = np.where(mnist.train.labels==i)
            label2images[i] = mnist.train.images[idx]

        images = []
        images_next = []
        for i in range(10):
            i_next = (i+1)%10
            k = min(label2images[i].shape[0], label2images[i_next].shape[0])
            images.append(label2images[i][:k])
            images_next.append(label2images[i_next][:k])
        self.train = DataSet(np.concatenate(images), np.concatenate(images_next))


class DataSet(object):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.images = X
        self.labels = y
        self._num_examples = X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._X = self.X[perm0]
            self._y = self.y[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            X_rest_part = self._X[start:self._num_examples]
            y_rest_part = self._y[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._X = self.X[perm]
                self._y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            X_new_part = self._X[start:end]
            y_new_part = self._y[start:end]
            return np.concatenate((X_rest_part, X_new_part), axis=0) , np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]


class DataSet_grid(object):
    def __init__(self,Xd,yd):
        self.Xd = Xd
        self.yd = yd
        self.keys = Xd.keys()
        self._num_keys = len(Xd.keys())
        self._num_examples = 0
        self._num_examplesd = {}
        for key in self.keys:
            self._num_examples += self.Xd[key].shape[0]
            self._num_examplesd[key] = self.Xd[key].shape[0]

        self._epochs_completed = 0
        self._index_in_epoch   = 0
        self._key_in_epoch     = 0
        self.x_dim             = self.Xd[key].shape[1]
        self.y_dim             = self.yd[key].shape[1]


    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examplesd[self._key_in_epoch])
            np.random.shuffle(perm0)
            self._X = self.Xd[self._key_in_epoch][perm0]
            self._y = self.yd[self._key_in_epoch][perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examplesd[self._key_in_epoch]:
            # Finished epoch
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examplesd[self._key_in_epoch] - start
            X_rest_part = self._X[start:self._num_examples]
            y_rest_part = self._y[start:self._num_examples]
            self._key_in_epoch += 1
            if self._key_in_epoch == self._num_keys:
                self._epochs_completed += 1
                self._key_in_epoch = 0
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examplesd[self._key_in_epoch])
                np.random.shuffle(perm)
                self._X = self.Xd[self._key_in_epoch][perm]
                self._y = self.yd[self._key_in_epoch][perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            X_new_part = self._X[start:end]
            y_new_part = self._y[start:end]

            return np.concatenate((X_rest_part, X_new_part), axis=0) , np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]



def parse_grid_data(robot_paths, goal_grids, obj_grids, num_per_path):
    keys = robot_paths.keys()
    grid_size = goal_grids[list(keys)[0]].shape[0]
    sz_y = grid_size*grid_size*3
    sz_x = grid_size*grid_size
    Xd = {}
    yd = {}
    X = np.zeros([0, sz_x])
    y = np.zeros([0, sz_y])

    count = 0
    d_ind = 0
    for key in keys:
        if count % 100 == 0:
            print(count)
        if (count % 200 == 0) & (count > 0):
            Xd[d_ind] = X
            yd[d_ind] = y
            d_ind += 1
            X = np.zeros([0, sz_x])
            y = np.zeros([0, sz_y])

        rob_path  = robot_paths[key]
        goal_grid = goal_grids[key]
        obj_grid  = obj_grids[key]

        num_steps = rob_path.shape[2]
        inds = []
        for i in range(num_steps):
            for j in range(i+1, num_steps):
                inds.append((i,j))

        random.shuffle(inds)

        for p in inds[:num_per_path]:
            i = p[0]
            j = p[1]
            y1 = rob_path[:,:,i].flatten()[None,:]
            y2 = goal_grid.flatten()[None,:]
            y3 = obj_grid.flatten()[None,:]
            y_t  = np.concatenate((y1,y2,y3), axis=1)
            X_t  = rob_path[:,:,j].flatten()[None, :]
            X = np.concatenate((X, X_t), axis=0)
            y = np.concatenate((y, y_t), axis=0)
        count += 1
    return(Xd, yd)


def parse_grid_data_test(robot_paths, goal_grids, obj_grids, num_per_path):
    keys = robot_paths.keys()
    grid_size = goal_grids[list(keys)[0]].shape[0]
    sz_y = grid_size*grid_size*3
    sz_x = grid_size*grid_size
    Xd = {}
    yd = {}
    X = np.zeros([0, sz_x])
    y = np.zeros([0, sz_y])

    count = 0
    d_ind = 0

    dict_size = min(len(keys),199)
    for key in keys:
        if count % 100 == 0:
            print(count)

        rob_path  = robot_paths[key]
        goal_grid = goal_grids[key]
        obj_grid  = obj_grids[key]

        num_steps = rob_path.shape[2]
        inds = []
        for i in range(num_steps - 1):
            inds.append((i,num_steps-1))

        #random.shuffle(inds)

        for p in inds[:num_per_path]:
            i = p[0]
            j = p[1]
            y1 = rob_path[:,:,i].flatten()[None,:]
            y2 = goal_grid.flatten()[None,:]
            y3 = obj_grid.flatten()[None,:]
            y_t  = np.concatenate((y1,y2,y3), axis=1)
            X_t  = rob_path[:,:,j].flatten()[None, :]
            X = np.concatenate((X, X_t), axis=0)
            y = np.concatenate((y, y_t), axis=0)
        count += 1
        if (count % 200 == dict_size):
            Xd[d_ind] = X
            yd[d_ind] = y
            d_ind += 1
            X = np.zeros([0, sz_x])
            y = np.zeros([0, sz_y])

    return(Xd, yd)

def parse_grid_data_fulltraj(robot_paths, goal_grids, obj_grids):
    keys = robot_paths.keys()
    grid_size = goal_grids[list(keys)[0]].shape[0]
    sz_y = grid_size*grid_size*3
    sz_x = grid_size*grid_size
    Xd = {}
    yd = {}
    X = np.zeros([0, sz_x])
    y = np.zeros([0, sz_y])

    count = 0
    d_ind = 0
    dict_size = min(len(keys),199)

    for key in keys:
        if count % 100 == 0:
            print(count)
        if (count % 200 == 0) & (count > 0):
            Xd[d_ind] = X
            yd[d_ind] = y
            d_ind += 1
            X = np.zeros([0, sz_x])
            y = np.zeros([0, sz_y])
        rob_path  = robot_paths[key]
        goal_grid = goal_grids[key]
        obj_grid  = obj_grids[key]
        num_steps = rob_path.shape[2]
        inds = []
        for i in range(num_steps-1):
            y1 = rob_path[:,:,i].flatten()[None,:]
            y2 = goal_grid.flatten()[None,:]
            y3 = obj_grid.flatten()[None,:]
            y_t  = np.concatenate((y1,y2,y3), axis=1)
            X_t  = rob_path[:,:,i+1].flatten()
            for j in range(i+2, num_steps):
                X_t += rob_path[:,:,j].flatten()
            X = np.concatenate((X, X_t[None,:]), axis=0)
            y = np.concatenate((y, y_t), axis=0)
        count += 1
        if (count % 200 == dict_size):
            Xd[d_ind] = X
            yd[d_ind] = y
            d_ind += 1
            X = np.zeros([0, sz_x])
            y = np.zeros([0, sz_y])

    return(Xd, yd)


def parse_sokoban_train_test(data, train_examples, test_examples):
    Xd_train, yd_train, ind = parse_sokoban_data(data,train_examples,0)
    Xd_test, yd_test, _ = parse_sokoban_data(data,test_examples,0)
    return(Xd_train, yd_train, Xd_test, yd_test)


def parse_sokoban_data(data, examples = 20000, start_ind = 0):
    robot_loc = data.get('robot_loc').value
    obj_loc   = data.get('obj_loc').value
    action    = data.get('action').value
    wall      = data.get('wall').value
    goal_loc  = data.get('goal_loc').value
    sequence_length = data.get('sequence_length').value
    sequence_mask   = data.get('sequence_mask').value
    grid_shape = data.attrs['world_shape']

    num_obj = obj_loc.shape[1]
    num_ex = robot_loc.shape[0]
    sz_y = grid_shape[0]**2*4
    sz_x = grid_shape[0]**2
    Xd = {}
    yd = {}
    X = np.zeros([0, sz_x])
    y = np.zeros([0, sz_y])
    d_ind = 0
    count = 0
    tot_ex = 0
    i = start_ind

    while tot_ex < examples:
        if count % 10 == 0:
            print(count)
        if (count % 10 == 0) & (count > 0):
            Xd[d_ind] = X
            yd[d_ind] = y
            d_ind += 1
            tot_ex += X.shape[0]
            X = np.zeros([0, sz_x])
            y = np.zeros([0, sz_y])


        mask = sequence_mask[i]
        length = sequence_length[i]
        robot_path = robot_loc[i][mask]
        obj_path   = obj_loc[i, 0][mask]
        obstacles  = wall[i].astype(int)
        goals      = goal_loc[i]
        goal_grid  = np.zeros(grid_shape)
        goal_grid[goals[:,0], goals[:,1]] = 1

        for i1 in range(length):
            robot_grid = np.zeros(grid_shape)
            robot_xy  = robot_path[i1]
            robot_grid[robot_xy[0], robot_xy[1]] = 1
            obj_grid   = np.zeros(grid_shape)
            obj_xy    = obj_path[i1]
            obj_grid[obj_xy[0], obj_xy[1]] = 1
            y1 = robot_grid.flatten()[None,:]
            y2 = obj_grid.flatten()[None,:]
            y3 = obstacles.flatten()[None,:]
            y4 = goal_grid.flatten()[None,:]
            y_t = np.concatenate((y1,y2,y3,y4), axis=1)
            for i2 in range(i1+1, length):
                nobj_grid   = np.zeros(grid_shape)
                nobj_xy     = obj_path[i2]
                nobj_grid[nobj_xy[0], nobj_xy[1]] = 1
                X_t = nobj_grid.flatten()[None,:]
                X = np.concatenate((X, X_t), axis=0)
                y = np.concatenate((y, y_t), axis=0)
        count += 1
        i += 1
    return(Xd, yd, i)





















