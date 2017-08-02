from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


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
