import tensorflow as tf
import numpy as np
import scipy.io as sio
import math

def GCN(data_dir, ratio):
    train_data = sio.loadmat(data_dir)
    x_train = train_data['X']
    y_train = train_data['y']
    y_train[y_train == 10] = 0
    x_train = x_train.transpose((3,0,1,2))
    x_train.astype(float)
    x_gray = np.dot(x_train, [[0.2989],[0.5870],[0.1140]])

    imsize = x_gray.shape[0]
    mean = np.mean(x_gray, axis=(1,2), dtype=float)
    std = np.std(x_gray, axis=(1,2), dtype=float, ddof=1)
    std[std < 1e-4] = 1
    x_GCN = np.zeros(x_gray.shape, dtype=float)
    for i in np.arange(imsize):
        x_GCN[i,:,:] = (x_gray[i,:,:] - mean[i]) / std[i]
    nums = x_GCN.shape[0]
    x_GCN = x_GCN.reshape((nums,-1))
    data = np.hstack((y_train,x_GCN))
    np.random.shuffle(data)
    cut=math.floor(nums*ratio)
    train,val = data[:cut,:], data[cut:,:]

    print("\n------- GCN done -------")
    return train, val



def read_SVHN(data_dir, ratio, batch_size):

    train,val = GCN(data_dir, ratio)
    img_width = 32
    img_height = 32
    img_depth = 1
    label_bytes = 1
    image_bytes = 1024
    record_bytes = 1025

    with tf.name_scope('input'):
        images_list = []
        label_batch_list = []
        for train_val in [train, val]:
            q = tf.train.input_producer(train_val)
            input_data = q.dequeue()

            label = tf.slice(input_data , [0], [1])
            label = tf.cast(label, tf.int32)

            image_raw = tf.slice(input_data , [1], [1024])
            image_raw = tf.reshape(image_raw, [1, 32, 32])
            image = tf.transpose(image_raw, (1,2,0))
            image = tf.cast(image, tf.float32)
            images, label_batch = tf.train.batch([image, label],
                                                batch_size = batch_size,
                                                num_threads = 16,
                                                capacity= 2000)

            n_classes = 10
            label_batch = tf.one_hot(label_batch, depth = n_classes)

            label_batch_list.append(tf.reshape(label_batch, [batch_size, n_classes]))
            images_list.append(images)


        return images_list, label_batch_list

if __name__ == "__main__":
    data_dir = './data/SVHN/train_32x32.mat'
    ratio = 0.1
    train, val = GCN(data_dir, ratio)
    print('train has the shape: {0} '.format(train.shape))
    print('val has the shape: {0} '.format(val.shape))
    print('Total has {0} records'.format(train.shape[0]+val.shape[0]))
