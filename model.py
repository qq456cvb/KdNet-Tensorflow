#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.tfutils import get_current_tower_context, summary
from dataset import  MyDataflow
import multiprocessing

# Monkey-patch tf.layers to support argscope.
enable_argscope_for_module(tf.layers)

DEPTH = 11
N_CLASSES = 40
DIM = 3


class Model(ModelDesc):
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, None, DIM), 'input'),
                tf.placeholder(tf.int32, (None, None, None), 'idx'),
                tf.placeholder(tf.int32, (None,), 'label'),
                *[tf.placeholder(tf.int32, (None, None), 'split_axis%d' % i) for i in range(DEPTH)]]

    def build_graph(self, points, idx, label, *split_axis):
        """This function should build the model which takes the input variables
        and return cost at the end"""

        # add all features in the leaf node
        batch_idx = tf.expand_dims(tf.tile(tf.reshape(tf.range(tf.shape(points)[0]), (-1, 1, 1)), [1, tf.shape(idx)[1], tf.shape(idx)[2]]), -1)
        points = tf.gather_nd(points, tf.concat([batch_idx, tf.expand_dims(idx, -1)], -1))
        points = tf.transpose(tf.reduce_mean(points, -2), (0, 2, 1))  # B * N * 3
        x = tf.transpose(tf.nn.conv1d(points, tf.get_variable('kernel_pre', [1, DIM, 32]), 1, 'SAME', data_format='NCHW'), (0, 2, 1))

        features = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 128, N_CLASSES]
        Ws = [tf.get_variable('kernel%d' % i, shape=(DIM, 2 * features[i], features[i + 1])) for i in range(DEPTH)]
        Bs = [tf.get_variable('bias%d' % i, shape=(DIM, features[i + 1])) for i in range(DEPTH)]
        for i in range(DEPTH):
            x = tf.expand_dims(tf.reshape(x, [tf.shape(x)[0], tf.div(tf.shape(x)[1], 2), 2 * features[i]]), 2)  # B * N/2 * 1 * 2F
            w = tf.gather_nd(Ws[i], tf.expand_dims(split_axis[i], -1))  # B * N/2 * 2F * F_next
            # x = tf.Print(x, [tf.shape(x), tf.shape(w)], summarize=100)
            b = tf.gather_nd(Bs[i], tf.expand_dims(split_axis[i], -1))  # B * N/2 * F_next
            x = tf.squeeze(tf.matmul(x, w), -2) + b
            if i < DEPTH - 1:
                x = tf.nn.relu(x)

        logits = tf.squeeze(x, 1)

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('kernel.*', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('kernel.*', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = 1e-3
        # lr = tf.train.exponential_decay(
        #     learning_rate=1e-3,
        #     global_step=get_global_step_var(),
        #     decay_steps=468 * 10,
        #     decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.MomentumOptimizer(lr, 0.9)


def get_data():
    train = PrefetchData(BatchData(MyDataflow('./modelnet40_ply_hdf5_2048/train_files.txt', False, True), 32), multiprocessing.cpu_count() // 2, multiprocessing.cpu_count() // 2)
    test = PrefetchData(BatchData(MyDataflow('./modelnet40_ply_hdf5_2048/test_files.txt', False, False), 32, remainder=True), multiprocessing.cpu_count() // 2, multiprocessing.cpu_count() // 2)
    return train, test


if __name__ == '__main__':
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset_train)

    # get the config which contains everything necessary in a training
    config = TrainConfig(
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=FeedInput(dataset_train),
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )
    launch_train_with_config(config, SimpleTrainer())
