import os
import os.path
import numpy as np
import tensorflow as tf
import tools
import model
import math
from datetime import datetime
import input_data

N_CLASSES = 10
IMG_W =  32 # resize the image, if the input image is too large, training will be very slow.
IMG_H = 32
RATIO = 0.2 # take 20% of dataset as validation data
BATCH_SIZE = 64
TEST_BATCH_SIZE = 20
CAPACITY = 20
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001



data_dir = './data/SVHN/train_32x32.mat'
bin_dir = './data/SVHN/'
train_log_dir = './logs/train/'
val_log_dir = './logs/val/'
def train():

    with tf.name_scope('input'):

        image_batch, label_batch  = input_data.read_SVHN(data_dir = data_dir,
                                                            ratio = 0.1,
                                                            batch_size = 64)
        tra_image_batch = image_batch[0]
        tra_label_batch = label_batch[0]

        val_image_batch = image_batch[1]
        val_label_batch = label_batch[1]




    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE,N_CLASSES])

    logits = model.SVHN(x, N_CLASSES)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)


    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)


        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x:tra_images, y_:tra_labels})
                if step % 50 == 0 or (step + 1) == MAX_STEP:

                    print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    _, summary_str = sess.run([train_op, summary_op], feed_dict={x: tra_images, y_: tra_labels})
                    tra_summary_writer.add_summary(summary_str, step)

                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images,y_:val_labels})

                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                    _, summary_str = sess.run([train_op, summary_op], feed_dict={x: val_images, y_: val_labels})
                    val_summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                    saver.save(sess,
                               checkpoint_path,
                               global_step=step,
                               write_meta_graph=False)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    train()
