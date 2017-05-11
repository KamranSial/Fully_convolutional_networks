from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

#VGG_MEAN = [103.939, 116.779, 123.68]
VGG_MEAN = [72.3900075701, 82.9080206596,73.1574881705 ]
DATA_STD = [45.3104437099, 46.1445214188, 44.906197822]

class FCN32VGG:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("Model Loaded")
        else:
            self.data_dict = None
        
        self.var_dict = {}
        self.wd = 5e-4
        self.freez=True

    def build(self, rgb, train=None, num_classes=19, random_init_fc8=True,
              debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):
            red, green, blue = tf.split(3,3,rgb)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(3,[
                (blue - VGG_MEAN[0])/DATA_STD[0],
                (green - VGG_MEAN[1])/DATA_STD[1],
                (red - VGG_MEAN[2])/DATA_STD[2],
            ])

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        self.conv1_1 = self._conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train is not None:
            self.fc6 = tf.cond(train, lambda: tf.nn.dropout(self.fc6, 0.5), lambda: self.fc6)
            print("fc6 dropout_added")

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train is not None:
            self.fc7 = tf.cond(train, lambda: tf.nn.dropout(self.fc7, 0.5), lambda: self.fc7)
            print("fc7 dropout_added")

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)

        self.upscore = self._upscore_layer(self.score_fr, shape=tf.shape(bgr),
                                           num_classes=num_classes,
                                           debug=debug,
                                           name='up', ksize=64, stride=32)

        self.pred_up = tf.argmax(self.upscore, dimension=3)
        self.data_dict = None

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(3, in_channels, out_channels,name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            
            conv_biases = self.get_bias(out_channels,name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            #_activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
                conv_biases = self.get_bias(4096,name, num_classes=num_classes)
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 19],
                                                  num_classes=num_classes)
                conv_biases = self.get_bias(19, name, num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
                conv_biases = self.get_bias(4096,name, num_classes=num_classes)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            #_activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            num_input = in_features
            stddev = (2 / num_input)**0.5
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            print('Layer name: %s' % name)
            print('Layer shape: %s' % str(shape))
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            #_activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.pack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        #_activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape,trainable=False)

    def get_conv_filter(self, filter_size, in_channels, out_channels, name):
        if self.data_dict is not None and name in self.data_dict:
            init = tf.constant_initializer(value=self.data_dict[name][0],
                                           dtype=tf.float32)
            shape = self.data_dict[name][0].shape
            print (name, "Value Loaded")
        else:
            init=tf.contrib.layers.xavier_initializer_conv2d()
            shape=[filter_size, filter_size, in_channels, out_channels]
            print (name, "2d Value Initialized")
        
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        if ((name=="conv1_1") and self.freez==True):
            var = tf.get_variable(name="filter", initializer=init, shape=shape, trainable=False)
            print(name + " Frozen")
        else:        
            var = tf.get_variable(name="filter", initializer=init, shape=shape)
        
        self.var_dict[(name, 0)] = var
        
        if not tf.get_variable_scope().reuse:
            if not ((name=="conv1_1") and self.freez==True):
                weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                      name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
                print ("L2 Loss Added")
        return var

    def get_bias(self,out_channels,name, num_classes=None):
        if self.data_dict is not None and name in self.data_dict:
            bias_wights = self.data_dict[name][1]
            shape = self.data_dict[name][1].shape
            #if name == 'fc8':
            #    bias_wights = self._bias_reshape(bias_wights, shape[0],
            #                                 num_classes)
            #    shape = [num_classes]
            init = tf.constant_initializer(value=bias_wights,dtype=tf.float32)
            var = tf.get_variable(name="biases", initializer=init, shape=shape)
            print (name, shape, "Value Loaded\n\n")
        else:
            var=tf.get_variable(name="biases",initializer=tf.zeros_initializer(shape=[out_channels],dtype=tf.float32))
            print (name, "Value Initialized 0\n\n")
        
        self.var_dict[(name, 1)] = var    
        
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`
        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.
        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.
        Consider reordering fweight, to perserve semantic meaning of the
        weights.
        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes
        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        if self.data_dict is not None:
            initializer = tf.truncated_normal_initializer(stddev=stddev)
            print ("fc8", "Value Randomly initialized")
        else:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            print ("fc8", " 2d Value xavier Initialized")
        var = tf.get_variable("weights", shape=shape,
                              initializer=initializer)
        self.var_dict[('fc8', 0)] = var
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            print ("L2 Loss Added")
        
        return var
        
    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        print ("fc8_biases", "Value Initialized 0\n\n")
        var=tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)
        self.var_dict[('fc8', 1)] = var
        return var
        

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        if self.data_dict is not None and name in self.data_dict:
            weights = self.data_dict[name][0]
            #weights = weights.reshape(shape)
            #if num_classes is not None:
            #    weights = self._summary_reshape(weights, shape,
            #                                    num_new=num_classes)
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            print (name, "Value Loaded")
        else:
            init=tf.contrib.layers.xavier_initializer_conv2d()
            print (name, "2d Value Initialized")
        var=tf.get_variable(name="weights", initializer=init, shape=shape)
        self.var_dict[(name, 0)] = var
        #if self.wd and (not tf.get_variable_scope().reuse):
        #    weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd, name='weight_loss')
        #    tf.add_to_collection('losses', weight_decay)
        #    print ("L2 Loss Added")
        return var

    def _activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        #tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        #tf.summary.histogram(tensor_name + '/activations', x)
        #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def save_npy(self, sess, npy_path="./fcn32-save.npy"):
        assert isinstance(sess, tf.Session)
        data_dict = {}
        for (name, idx), var in self.var_dict.items():
            #print (name,idx,var.get_shape())
            var_out = sess.run(var)
            #print(var_out)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path