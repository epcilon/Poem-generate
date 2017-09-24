#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from rhyme import RhymeDict
from word2vec import *
from data_utils import *
from collections import deque
import tensorflow as tf
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_model_path = os.path.join(save_dir, 'classifiermodel')

_BATCH_SIZE = 10
_LABLE_NUM = 2



class NameClassfier:
    def __init__(self):
        embedding = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE, NUM_UNITS]), trainable=False)
        self._embed_ph = tf.placeholder(tf.float32, [VOCAB_SIZE, NUM_UNITS])
        self._embed_init = embedding.assign(self._embed_ph)


        self.cell_fw = tf.contrib.rnn.LSTMCell(
                          NUM_UNITS,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                          state_is_tuple=False)
        self.cell_bw = tf.contrib.rnn.LSTMCell(
                          NUM_UNITS,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                          state_is_tuple=False)

        self.init_fw_state = self.cell_fw.zero_state(_BATCH_SIZE, dtype=tf.float32)
        self.init_bw_state = self.cell_bw.zero_state(_BATCH_SIZE, dtype=tf.float32)
        self.inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.lengths = tf.placeholder(tf.int32, [_BATCH_SIZE])


        outputs,self.final_fw_state,self.final_bw_state = tf.contrib.rnn.static_bidirectional_rnn(
                          self.cell_fw, self.cell_bw, self.inputs, dtype=tf.float32,
                          sequence_length=self.lengths)

        softmax_w = tf.get_variable('softmax_w', [NUM_UNITS, _LABLE_NUM])
        softmax_b = tf.get_variable('softmax_b', [_LABLE_NUM])

        logits = tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, NUM_UNITS]), softmax_w),
                                bias=softmax_b)
        self.probs = tf.nn.softmax(logits)

        self.targets = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        labels = tf.one_hot(tf.reshape(self.targets, [-1]), depth=_LABLE_NUM)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits,
                            labels=labels)
        self.loss = tf.reduce_mean(loss)

        self.learn_rate = tf.Variable(0.0, trainable=False)
        self.opt_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())
        self.int2ch, self.ch2int = get_vocab()

        def _init_vars(self, sess):
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                sess.run(init_op)
                sess.run([self._embed_init], feed_dict={
                    self._embed_ph: get_word_embedding(NUM_UNITS)})
            else:
                self.saver.restore(sess, ckpt.model_checkpoint_path)

        def _train_a_batch(self, sess, mats, lens, target):
            total_loss = 0

            fw_state = self.init_fw_state
            bw_state = self.init_bw_state

            fw_state,bw_state, loss, _ = sess.run([self.decoder_final_state, self.loss, self.opt_op], feed_dict={
                self.init_fw_state: fw_state,
                self.init_bw_state: bw_state,
                self.inputs: mats,
                self.lengths: lens,
                self.targets: target})
            total_loss += loss
            print "loss = %f" % (total_loss)