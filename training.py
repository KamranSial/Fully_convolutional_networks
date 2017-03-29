from __future__ import print_function
import tensorflow as tf
import numpy as np

import datetime
import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow as tf

import os
import time
import numpy as np
from IPython.display import clear_output
import sys

import fcn32_vgg
import utils
ckpt_dir = "/home/sik4hi/ckpt_dir"
LOGS_PATH = '/home/sik4hi/tensorflow_logs'
WEIGHT_PATH = '.npy'
TRAINSET_PATH = '/mnt/data1/city/csv_files/cityscapes_train.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
IMAGE_SIZE = 224
NUM_OF_CLASSESS = 19
BATCH_SIZE = 1
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
NUM_CHANNELS = 3


csv_path = tf.train.string_input_producer([TRAINSET_PATH],
                                              shuffle=True)
textReader = tf.TextLineReader()
_, csv_content = textReader.read(csv_path)
im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

im_content = tf.read_file(im_name)
train_image = tf.image.decode_png(im_content, channels=3)
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

la_content = tf.read_file(im_label)
label_image = tf.image.decode_png(la_content, channels=1)
label_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
label_image=tf.squeeze(label_image, squeeze_dims=[2])
label_image = tf.one_hot(label_image,19)

train_image = tf.cast(train_image, tf.float32) / 255.
image = tf.Print(train_image, [tf.shape(label_image)])
train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, label_image], batch_size=BATCH_SIZE,
                                                               capacity=3 + 3 * BATCH_SIZE,
                                                               min_after_dequeue=3)
with tf.device('/cpu:0'):

    sess = tf.Session()
    images_tf = tf.placeholder(tf.float32, [None, 1024, 2048, 3])
    labels_tf = tf.placeholder(tf.float32,[None, 1024, 2048, 19])

    vgg_fcn = fcn32_vgg.FCN32VGG('./vgg16.npy')
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images_tf, train=True, num_classes=19, random_init_fc8=True, debug=False)


    #head=[]
    cross_entropy = -tf.reduce_sum(
        labels_tf * tf.log(vgg_fcn.softmax), reduction_indices=[1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    #loss_tf = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(vgg_fcn.upscore,
     #                                                                     tf.squeeze(labels_tf, squeeze_dims=[3]),
      #                                                                    name="entropy")))

    train_op = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy_mean)

    print('Finished building Network.')
    init_op = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())
    sess.run(init_op)
    # print(csv_path)
    # For populating queues with batches, very important!
    # coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(1):
        train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
        print("batch made")
        _, train_loss = sess.run([train_op, cross_entropy_mean],
            feed_dict={images_tf: train_imbatch, labels_tf: train_labatch})
        #train_loss= sess.run(vgg_fcn.upscore,
         #   feed_dict={ images_tf: train_imbatch})
        #res=sess.run(image)
        #plt.imshow(train_labatch[0])
        #plt.show()
        print ("Training Loss:", train_loss)
        #print(len(train_labatch[0]))
        #print((train_labatch[0][0]))
        #print(train_labatch[0][j][k])
        #print(train_1labatch[0][j][k])

    #print(train_imbatch[0])
    print("training finished")