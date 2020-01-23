import numpy as np
import random
import scipy.io


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    if mode == 'train':

        x_train = scipy.io.loadmat('train_images_14bus.mat')['x_train']
        y_train = scipy.io.loadmat('train_labels_14bus.mat')['y_train']

        x_valid = scipy.io.loadmat('valid_images_14bus.mat')['x_valid']
        y_valid = scipy.io.loadmat('valid_labels_14bus.mat')['y_valid']
        print(x_valid.shape, y_valid.shape, type(x_valid), type(y_valid))

        x_train, _ = reformat(x_train, y_train)
        x_valid, _ = reformat(x_valid, y_valid)
        print(x_valid.shape, y_valid.shape, type(x_valid), type(y_valid))

        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test = scipy.io.loadmat('test_images_14bus.mat')['x_test']
        y_test = scipy.io.loadmat('test_labels_14bus.mat')['y_test']
        x_test, _ = reformat(x_test, y_test)
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param y: corresponding labels
    :return: reshaped input and labels
    """
    # img_size, num_ch, num_class = int(np.sqrt(x.shape[1])), 1, len(np.unique(np.argmax(y, 1)))
    img_size, num_ch, num_class = 14, 1, 16
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)  # =[1 2 3 ... 10]??
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch
