#!/usr/bin/dev python
#coding=utf-8
#author: spgoal
#data: 2018-02-09

import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import json
from multiprocessing import Pool

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

# constant numbers
BATCH_SIZE = 1 << 6
DATA_SIZE = 36
DATA_HEIGHT = 36
DATA_WIDTH = 16
OUTPUT_TYPE = 5
KEEP_PROBS = [0.8, 0.8, 1.0, 1.0, 1.0]

class LotteryCNN():
  def __init__(self, train_type=0):
    # init constant
    self.data_height = DATA_HEIGHT
    self.data_width = DATA_WIDTH
    self.batch_size = BATCH_SIZE
    self.keep_probs = KEEP_PROBS
    self.adam_params = [1e-5, 1e-5, 1e-4, 1e-4, 1e-4]
    self.train_type = train_type
    self.check_type_list = [3, 3, 9, 8, 31]
    # save graph dir
    self.graph_dir = os.path.join(cur_dir, "graph_output")
    if not os.path.exists(self.graph_dir):
      os.makedirs(self.graph_dir)
    # save cnn output dir
    self.cnn_dir = os.path.join(cur_dir, "cnn_output")
    if not os.path.exists(self.cnn_dir):
      os.makedirs(self.cnn_dir)
    # save history accuracy
    self.his_path = os.path.join(cur_dir, "data/history_accuracy_cnn.txt")
    self.load_history_accuracy()
    # load data
    self.data_file_path = os.path.join(cur_dir, "data/total_data.txt")
    self.np_data, self.np_labels = self.load_data(self.data_file_path)
    self.split_data()
    # construct cnn
    self.build_graph()

  def load_history_accuracy(self):
    self.his_acc = [0, 0, 0, 0, 0]
    if os.path.getsize(self.his_path) <= 0:
      return
    with open(self.his_path, "r") as rf:
      self.his_acc = json.load(rf)

  def strip_data(self, data):
    for x in data:
      x[0] = float(x[0]) / 1000
      x[1] = float(x[1]) / 1000
      x[2:9] = [0] * 7 
    return data

  def load_data(self, infile):
    rf = open(infile, "r")
    data_list = json.load(rf)
    self.data_len = len(data_list)
    print "self.data_len:", self.data_len
    rf.close()
    data = [x[0] for x in data_list]
    labels = [x[1] for x in data_list]
    np_data = np.array(data)
    np_data = self.strip_data(np_data)
    np_data = np.reshape(np_data, [-1, self.data_height, self.data_width, 1])
    zero_data = np.zeros([self.data_len, self.data_height, self.data_height - self.data_width, 1])
    np_data = np.concatenate([np_data, zero_data], axis=2)
    np_labels = np.array(labels)
    return np_data, np_labels

  def split_data(self):
    self.res_data = []
    counts = [9.5/10, 0.25/10, 0.25/10]
    shuf = np.random.permutation(np.arange(self.data_len))
    base_num = 0
    for pro in counts:
      bnd = int(self.data_len * pro)
      cur_data = []
      for i in range(bnd):
        cur_data.append((self.np_data[shuf[base_num + i], :, :, :], self.np_labels[shuf[base_num + i], self.train_type]))
      self.res_data.append(cur_data)
      base_num += bnd
    self.train_data = self.res_data[0]
    self.dev_data = self.res_data[1]
    self.test_data = self.res_data[2]
    print "len of data size, (train, dev, test): (%d, %d, %d)" % (len(self.train_data), len(self.dev_data), len(self.test_data))

  def get_batch(self, data_set, rank):
    data_len = len(data_set)
    data, label = [], []
    for dd, ll in data_set:
      data.append(dd)
      label.append(ll)
    beg = (rank * self.batch_size) % data_len
    end = ((rank + 1) * self.batch_size) % data_len
    if beg < end:
      x_out, y_out = data[beg : end], label[beg : end]
    else:
      x_out = data[beg : data_len]
      y_out = label[beg : data_len]
      x_out.extend(data[0 : end])
      y_out.extend(label[0 : end])
      x_out, y_out = x_out, y_out
    return np.asarray(x_out), np.asarray(y_out)

  def get_random_batch(self, data_set, batch_size):
    data_len = len(data_set)
    data, label = [], []
    for dd, ll in data_set:
      data.append(dd)
      label.append(ll)
    shuf = np.random.permutation(np.arange(data_len))[0 : batch_size]
    x_out, y_out = [], []
    for i in shuf:
      x_out.append(data[i])
      y_out.append(label[i])
    return np.asarray(x_out), np.asarray(y_out)

  def build_graph(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.input_x = tf.placeholder(tf.float32, [None, self.data_height, self.data_height, 1], name="input_x")
      self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
      size_y = self.check_type_list[self.train_type]
      print "build graph with size y:", size_y
      self.prob_holder = tf.placeholder(tf.float32, name="keep_prob")
      self.build_cnn_layer(self.input_x, self.input_y, size_y)

  def build_cnn_layer(self, input_x, input_y, size_y):
    channels = [1 << 5, 1 << 6, 1 << 7, 1 << 5]
    with tf.name_scope("layer_1"):
      # conved (-1, 36, 36, 1) -> (-1, 32, 32, 32)
      w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, channels[0]], stddev=0.1), name="w_conv1")
      b_conv1 = tf.Variable(tf.constant(0.1, shape=[channels[0]]), name="b_conv1")
      v_conv1 = tf.nn.conv2d(input_x, w_conv1, strides=[1, 1, 1, 1], padding="VALID", name="v_conv1")
      h_conv1 = tf.nn.relu(v_conv1 + b_conv1, name="h_conv1")
      # pooled (-1, 32, 32, 32) -> (-1, 16, 16, 32)
      h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
      # batch normalization
      #fc_mean, fc_var = tf.nn.moments(h_pool1, axes=[0])
      #scale = tf.Variable(tf.ones([1]))
      #shift = tf.Variable(tf.zeros([1]))
      #epsilon = 0.001
      #h_norm1 = tf.nn.batch_normalization(h_pool1, fc_mean, fc_var, scale, shift, epsilon, name="norm1")
      h_out1 = h_pool1
    with tf.name_scope("layer_2"):
      # conv: (-1, 16, 16, 32) -> (-1, 14, 14, 64)
      w_conv2 = tf.Variable(tf.truncated_normal([3, 3, channels[0], channels[1]], stddev=0.1), name="w_conv2")
      b_conv2 = tf.Variable(tf.constant(0.1, shape=[channels[1]]), name="b_conv2")
      v_conv2 = tf.nn.conv2d(h_out1, w_conv2, strides=[1, 1, 1, 1], padding="VALID", name="v_conv2")
      h_conv2 = tf.nn.relu(v_conv2 + b_conv2, name="h_conv2")
      # pool: (-1, 14, 14, 64) -> (-1, 7, 7, 64)
      h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
      # batch normalization
      #fc_mean, fc_var = tf.nn.moments(h_pool2, axes=[0])
      #scale = tf.Variable(tf.ones([1]))
      #shift = tf.Variable(tf.zeros([1]))
      #epsilon = 0.001
      #h_norm2 = tf.nn.batch_normalization(h_pool2, fc_mean, fc_var, scale, shift, epsilon, name="norm2")
      h_out2 = h_pool2
    with tf.name_scope("layer_3"):
      # conv: (-1, 7, 7, 64) ->  (-1, 5, 5, 256)
      w_conv3 = tf.Variable(tf.truncated_normal([3, 3, channels[1], channels[2]], stddev=0.1), name="w_conv3") 
      b_conv3 = tf.Variable(tf.constant(0.1, shape=[channels[2]]), name="b_conv3")
      v_conv3 = tf.nn.conv2d(h_out2, w_conv3, strides=[1, 1, 1, 1], padding="SAME", name="v_conv3")
      h_conv3 = tf.nn.relu(v_conv3 + b_conv3)
      # pool: (-1, 5, 5, 256) -> (-1, 3, 3, 256)
      h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
      # batch_normaliztion
      #fc_mean, fc_var = tf.nn.moments(h_pool3, axes=[0])
      #scale = tf.Variable(tf.ones([1]))
      #shift = tf.Variable(tf.zeros([1]))
      #epsilon = 0.001
      #h_norm3 = tf.nn.batch_normalization(h_pool3, fc_mean, fc_var, scale, shift, epsilon, name="norm3")
      h_out3 = h_pool3
    # (-1, 3, 3, 256) -> (-1, 32)
    with tf.name_scope("full_connect_1"):
      h_flat = tf.reshape(h_out3, [-1, 3*3*channels[2]])
      w_fc1 = tf.Variable(tf.truncated_normal([3*3*channels[2], channels[3]], stddev=0.1), name="w_fc")
      b_fc1 = tf.Variable(tf.constant(0.1, shape=[channels[3]]), name="b_fc")
      v_fc1 = tf.matmul(h_flat, w_fc1, name="v_fc1")
      h_fc1 = tf.nn.relu(v_fc1 + b_fc1, name="h_fc1")
      h_drop = tf.nn.dropout(h_fc1, self.prob_holder, name="h_drop")
    # (-1, 32) -> (-1, size_y)
    with tf.name_scope("full_connect_2"):
      w_fc2 = tf.Variable(tf.truncated_normal([channels[3], size_y], stddev=0.1), name="w_fc2")
      b_fc2 = tf.Variable(tf.constant(0.1, shape=[size_y]), name="b_fc2")
      v_fc2 = tf.matmul(h_drop, w_fc2, name="v_fc2")
      h_fc2 = tf.nn.relu(v_fc2 + b_fc2, name="h_fc2")
    with tf.name_scope("loss"):
      cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=h_fc2, name="cross")
      self.loss = tf.reduce_mean(cross, name="loss")
      tf.summary.scalar("loss", self.loss)
    with tf.name_scope("train"):
      self.adam_opt = tf.train.AdamOptimizer(self.adam_params[self.train_type], name="adam")
      self.train_step = self.adam_opt.minimize(self.loss, name="train_step")
    with tf.name_scope("output"):
      self.prediction = tf.argmax(h_fc2, 1, name="prediction")
      self.probability = tf.nn.softmax(h_fc2, name="probability")
      self.correct_prediction = tf.equal(tf.cast(self.input_y, tf.int32), tf.cast(self.prediction, tf.int32))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("accuracy", self.accuracy)
    # do saver work
    self.saver = tf.train.Saver()
    out_dir = os.path.join(cur_dir, "only_cnn_output/model_%s/" % self.train_type + time.strftime("%y%m%d", time.localtime()))
    self.ckpt_dir = out_dir + '/model.ckpt'
    if not os.path.exists(out_dir):
      os.makedirs(self.ckpt_dir)
    # do summary
    self.merged = tf.summary.merge_all()

  def train(self):
    with tf.Session(graph=self.graph) as sess:
      data_writer = tf.summary.FileWriter("logs/cnn", sess.graph)
      sess.run(tf.global_variables_initializer())
      train_times = 1000000
      loop = 1
      loss, acc = 0.0, 0.0
      for i in range(train_times):
        x_batch, y_batch = self.get_batch(self.train_data, i)
        for j in range(loop):
          summary, _, loss, acc = sess.run([self.merged, self.train_step, self.loss, self.accuracy], feed_dict={self.input_x: x_batch, self.input_y: y_batch, self.prob_holder: self.keep_probs[self.train_type]})
          data_writer.add_summary(summary, i * loop + j)
        cur_step = (i + 1) * loop
        if cur_step % 100 == 0:
          print "current step:", cur_step
          print "train loss: %g, accuracy: %g" % (loss, acc)
          x_batch, y_batch = self.get_random_batch(self.dev_data, 1024)
          loss, acc, prob = sess.run([self.loss, self.accuracy, self.probability], feed_dict={self.input_x: x_batch, self.input_y: y_batch, self.prob_holder: self.keep_probs[self.train_type]})
          print "tests loss: %g, accuracy: %g" % (loss, acc)
          take_down_pro = 0.001
          if self.his_acc[self.train_type] <= acc and take_down_pro * train_times * loop < cur_step:
            self.saver.save(sess, self.ckpt_dir)
            self.his_acc[self.train_type] = acc.item()
      with open(self.his_path, "w") as of:
        json.dump(self.his_acc, of)

def main_proc(num=0, use_gpus="0"):
  lc = LotteryCNN(train_type=num, use_gpus=use_gpus)
  lc.train()

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "7"
  #lc = LotteryCNN(0)
  #lc.train()
  for i in [0, 1]:
    lc = LotteryCNN(i)
    lc.train()
