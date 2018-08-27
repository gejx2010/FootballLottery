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
BATCH_SIZE = 1 << 7
DATA_SIZE = 36
DATA_HEIGHT = 36
DATA_WIDTH = 16
OUTPUT_TYPE = 5
KEEP_PROBS = [[1.0, 1.0], 
              [1.0, 1.0],
              [1.0, 1.0],
              [1.0, 1.0],
              [1.0, 1.0]]

class LotteryCNN():
  def __init__(self, train_type=0, use_gpus="2, 3"):
    # init constant
    self.data_height = DATA_HEIGHT
    self.data_width = DATA_WIDTH
    self.batch_size = BATCH_SIZE
    self.keep_probs = KEEP_PROBS
    self.adam_params = [1e-5, 1e-5, 1e-5, 1e-5, 1e-4]
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
    self.his_path = os.path.join(cur_dir, "history_accuracy.txt")
    self.his_acc = self.load_history_accuracy()
    # load data
    self.match_file_path = os.path.join(cur_dir, "total_match_result.txt")
    self.shouzhu_path = os.path.join(cur_dir, "shouzhu_data.txt")
    with open(self.match_file_path, "r") as rf:
      match_list = json.load(rf)
      print "Length of match list:", len(match_list)
    self.data_file_path = os.path.join(cur_dir, "total_data.txt")
    self.data_bcp_path = os.path.join(cur_dir, "backup_data.txt")
    # averaget the train data
    #self.train_data = self.get_data_with_type(self.train_data, self.train_type)
    # construct cnn
    # RL parameters
    self.epsilon = 0.9
    self.gamma = 0.9
    self.alpha = 0.1
    # build graph
    self.use_gpus = use_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = self.use_gpus
    self.build_graph()
    self.build_train_params()

  def load_history_accuracy(self):
    his_acc = [0, 0, 0, 0, 0]
    if os.path.getsize(self.his_path) <= 0:
      return his_acc
    with open(self.his_path, "r") as rf:
      his_acc = json.load(rf)
    return his_acc

  def load_data(self, infile):
    if os.path.getsize(infile) <= 0:
      return [], []
    rf = open(infile, "r")
    data_list = json.load(rf)
    self.data_len = len(data_list)
    rf.close()
    data = [x[0] for x in data_list]
    labels = [x[1] for x in data_list]
    np_data = np.array(data)
    np_data = np.reshape(np_data, [-1, self.data_height, self.data_width, 1])
    zero_data = np.zeros([self.data_len, self.data_height, self.data_height - self.data_width, 1])
    np_data = np.concatenate([np_data, zero_data], axis=2)
    np_data = np_data.__truediv__(1000)
    np_labels = np.array(labels)
    return np_data, np_labels

  def split_data(self):
    self.res_data = []
    counts = [5.0/7, 1.0/7, 1.0/7]
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

  def get_data_with_type(self, data, train_type):
    if train_type == 4:
      return data
    data_type = [list() for x in range(self.check_type_list[train_type])]
    for dd, ll in data:
      data_type[ll].append(dd.tolist())
    max_len = max([len(x) for x in data_type])
    print "with data type: %d, max len: %d" % (self.train_type, max_len)
    # take limit to min size of data
    if max_len <= self.batch_size:
      return data
    # get the limit train data
    res_data = []
    for dd in data_type:
      print "len of each dict:", len(dd)
      rd, len_dd = [], 0
      while len_dd < max_len:
        rd.extend(dd)
        len_dd = len(rd)
      rd = rd[0 : max_len]
      res_data.extend(rd)
    # shuf on res_data
    len_res = len(res_data)
    ans_data = []
    print "max_len:", max_len
    for i in range(max_len):
      base = i
      while base < len_res:
        ans_data.append((res_data[base], int(base / max_len)))
        base += max_len
    print "len of final data:", len(ans_data)
    return np.asarray(ans_data)

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

  def get_random_batch(self, data_set):
    data_len = len(data_set)
    data, label = [], []
    for dd, ll in data_set:
      data.append(dd)
      label.append(ll)
    shuf = np.random.permutation(np.arange(data_len))[0 : self.batch_size]
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
      self.size_y = size_y
      print "build graph with size y:", size_y
      self.prob_holder = tf.placeholder(tf.float32, name="keep_prob")
      self.eval_logits = self.build_cnn_layer(self.input_x, self.input_y, size_y, scope="eval_net", collect="eval_net_params")
      self.next_logits = self.build_cnn_layer(self.input_x, self.input_y, size_y, scope="target_net", collect="target_net_params")
      self.target_logits = tf.placeholder(tf.float32, [None, size_y], name="target_logits")
      t_params = tf.get_collection("target_net_params")
      e_params = tf.get_collection("eval_net_params")
      self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

  def batch_normal(self, hh, axes, name="norm", collect=None):
      # batch normalization
      fc_mean, fc_var = tf.nn.moments(hh, axes=axes, keep_dims=True)
      if collect:
        scale = tf.Variable(tf.ones([1]), collections=collect)
        shift = tf.Variable(tf.zeros([1]), collections=collect)
      else:
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
      epsilon = 0.001
      h_norm = tf.nn.batch_normalization(hh, fc_mean, fc_var, scale, shift, epsilon, name=name)
      return h_norm

  def build_cnn_layer(self, input_x, input_y, size_y, scope="CNN", collect="CNN"):
    with tf.name_scope(scope):
      with tf.name_scope("layer_1"):
        c_names = [collect, tf.GraphKeys.GLOBAL_VARIABLES]
        # conved (-1, 36, 36, 1) -> (-1, 34, 34, 512)
        layer_1_channel = 256
        w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, layer_1_channel], stddev=0.1), name="w_conv1", collections=c_names)
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[layer_1_channel]), name="b_conv1", collections=c_names)
        v_conv1 = tf.nn.conv2d(input_x, w_conv1, strides=[1, 1, 1, 1], padding="VALID", name="v_conv1")
        h_conv1 = tf.nn.relu(v_conv1 + b_conv1, name="h_conv1")
        #h_conv1 = tf.nn.sigmoid(v_conv1 + b_conv1, name="h_conv1")
        # pooled (-1, 34, 34, 512) -> (-1, 17, 17, 512)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        # batch normalization
        #h_norm1 = self.batch_normal(h_pool1, axes=[1, 2], name="norm1", collect=collect)
        h_out1 = h_pool1
      with tf.name_scope("layer_2"):
        # conv: (-1, 17, 17, 256) -> (-1, 15, 15, 512)
        layer_2_channel = 512
        w_conv2 = tf.Variable(tf.truncated_normal([3, 3, layer_1_channel, layer_2_channel], stddev=0.1), name="w_conv2", collections=c_names)
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[layer_2_channel]), name="b_conv2", collections=c_names)
        v_conv2 = tf.nn.conv2d(h_out1, w_conv2, strides=[1, 1, 1, 1], padding="VALID", name="v_conv2")
        h_conv2 = tf.nn.relu(v_conv2 + b_conv2, name="h_conv2")
        #h_conv2 = tf.nn.sigmoid(v_conv2 + b_conv2, name="h_conv2")
        # pool: (-1, 15, 15, 2048) -> (-1, 8, 8, 2048)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        # batch normalization
        #h_norm2 = self.batch_normal(h_pool2, axes=[1, 2], name="norm2", collect=collect)
        h_out2 = h_pool2
      with tf.name_scope("layer_3"):
        # conv: (-1, 8, 8, 512) ->  (-1, 6, 6, 1024)
        layer_3_channel = 1024
        w_conv3 = tf.Variable(tf.truncated_normal([3, 3, layer_2_channel, layer_3_channel], stddev=0.1), name="w_conv3", collections=c_names) 
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[layer_3_channel]), name="b_conv3", collections=c_names)
        v_conv3 = tf.nn.conv2d(h_out2, w_conv3, strides=[1, 1, 1, 1], padding="VALID", name="v_conv3")
        h_conv3 = tf.nn.relu(v_conv3 + b_conv3, name="h_conv3")
        #h_conv3 = tf.nn.sigmoid(v_conv3 + b_conv3, name="h_conv3")
        # pool: (-1, 6, 6, 4096) -> (-1, 3, 3, 4096)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
        # batch_normaliztion
        #h_norm3 = self.batch_normal(h_pool3, axes=[1, 2], name="norm3", collect=collect)
        h_out3 = h_pool3
      # (-1, 3, 3, 4096) -> (-1, 2048)
      with tf.name_scope("full_connect_1"):
        h_flat1 = tf.reshape(h_out3, [-1, 3*3*layer_3_channel])
        w_fc1 = tf.Variable(tf.truncated_normal([3*3*layer_3_channel, 2048], stddev=0.1), name="w_fc1", collections=c_names)
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]), name="b_fc1", collections=c_names)
        v_fc1 = tf.matmul(h_flat1, w_fc1, name="v_fc1")
        h_fc1 = tf.nn.relu(v_fc1 + b_fc1, name="h_fc1")
        #h_fc1 = tf.nn.sigmoid(v_fc1 + b_fc1, name="h_fc1")
        h_drop1 = tf.nn.dropout(h_fc1, self.prob_holder[0], name="h_drop1")
      # (-1, 2048) -> (-1, 128)
      with tf.name_scope("full_connect_2"):
        w_fc2 = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1), name="w_fc2", collections=c_names)
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b_fc2", collections=c_names)
        v_fc2 = tf.matmul(h_drop1, w_fc2, name="v_fc2")
        h_fc2 = tf.nn.relu(v_fc2 + b_fc2, name="h_fc2")
        #h_fc2 = tf.nn.sigmoid(v_fc2 + b_fc2, name="h_fc2")
        h_drop2 = tf.nn.dropout(h_fc2, self.prob_holder[1], name="h_drop2")
      # (-1, 128) -> (-1, size_y)
      with tf.name_scope("full_connect_3"):
        w_fc3 = tf.Variable(tf.truncated_normal([128, size_y], stddev=0.1), name="w_fc3", collections=c_names)
        b_fc3 = tf.Variable(tf.constant(0.1, shape=[size_y]), name="b_fc3", collections=c_names)
        v_fc3 = tf.matmul(h_drop2, w_fc3, name="v_fc3")
        h_fc3 = tf.add(v_fc3, b_fc3, name="h_fc3")
        return h_fc3

  def choose_action(self, out_pre):
    if self.epsilon < np.random.uniform():
      ran = np.random.choice(self.size_y, self.batch_size)
      return tf.one_hot(ran, depth = self.size_y, name="random_generate")
    else:
      return out_pre

  def compute_target_diff(self, tar_log, pre_log):
    max_idx = tf.argmax(tar_log, -1)
    max_idx_mat = tf.one_hot(max_idx, depth=self.size_y)
    # compute predict matrix
    pre_mat = tf.multiply(pre_log, max_idx_mat)
    # compute target matrix
    tar_mat = tf.multiply(tar_log, max_idx_mat)
    # return final difference
    self.fin_diff = tf.subtract(tar_mat, pre_mat)
    return self.fin_diff

  def compute_target_predict_diff(self, tar, tar_log, pre_log):
    # compute reward
    max_idx = tf.argmax(tar_log, -1)
    max_idx_mat = tf.one_hot(max_idx, depth=self.size_y)
    tar_mat = tf.one_hot(tar, depth=self.size_y)
    rew_mat = tf.multiply(tar_mat, max_idx_mat)
    rew_list = tf.reduce_max(rew_mat, -1)
    b_con = tf.constant([1.0])
    ant_list = 10 * tf.subtract(rew_list, b_con)
    # compute target matrix
    max_tar = tf.multiply(tar_log, max_idx_mat)
    tar_list = tf.reduce_max(max_tar, -1)
    # compute predict matrix
    max_pre = tf.multiply(pre_log, max_idx_mat)
    pre_list = tf.reduce_max(max_pre, -1)
    self.fin_diff = tf.subtract(ant_list + self.gamma * tar_list, pre_list)
    return self.fin_diff

  def compute_target_predict_diff_along_y(self, tar, tar_log, pre_log):
    tar_mat = tf.one_hot(tar, depth=self.size_y)
    #b_con = tf.constant([0.5])
    #rediff_tar_mat = (tar_mat - b_con) * 2
    tar_log_mat = tf.multiply(tar_mat, tar_log)
    pre_log_mat = tf.multiply(tar_mat, pre_log)
    rew_mat = tar_mat + self.gamma * tar_log
    self.fin_diff = tf.subtract(rew_mat, pre_log)
    return self.fin_diff

  def build_train_params(self):
    with self.graph.as_default():
      with tf.name_scope("orig_output"):
        self.orig_prediction = tf.argmax(self.eval_logits, 1, name="orig_prediction")
        self.orig_probability = tf.nn.softmax(self.eval_logits, name="orig_probability")
        self.orig_correct_prediction = tf.equal(tf.cast(self.input_y, tf.int32), tf.cast(self.orig_prediction, tf.int32), name="orig_correct_prediction")
        self.accuracy = tf.reduce_mean(tf.cast(self.orig_correct_prediction, tf.float32), name="orig_accuracy")
      self.pure_logits = self.choose_action(self.eval_logits)
      with tf.name_scope("output"):
        self.prediction = tf.argmax(self.pure_logits, 1, name="prediction")
        self.tar_prediction = tf.argmax(self.target_logits, 1, name="target_prediction")
        self.probability = tf.nn.softmax(self.pure_logits, name="probability")
        self.correct_prediction = tf.equal(tf.cast(self.input_y, tf.int32), tf.cast(self.prediction, tf.int32), name="correct_prediction")
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
        tf.summary.scalar("accuracy_%d" % self.train_type, self.accuracy)
      with tf.name_scope("loss"):
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.pure_logits)
        self.res_cor_pred = tf.reshape(tf.cast(self.correct_prediction, tf.float32), [-1, 1], name="reshape_correct_prediction")
        #self.reward = tf.reduce_mean(cross) * -10
        #self.q_tar = self.reward + self.gamma * self.target_logits
        self.diff_mat = self.alpha * self.compute_target_predict_diff_along_y(self.input_y, self.target_logits, self.pure_logits)
        #diff_cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.diff_mat)
        self.loss = tf.reduce_mean(tf.square(self.diff_mat), name="loss")
        tf.summary.scalar("loss_%d" % self.train_type, self.loss)
      with tf.name_scope("train"):
        #self.adam_opt = tf.train.AdamOptimizer(self.adam_params[self.train_type], name="adam")
        #self.train_step = self.adam_opt.minimize(self.loss, name="train_step")
        self._lr = tf.Variable(0.0, trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        self.train_step = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
      # do saver work
      self.saver = tf.train.Saver()
      # do summary
      self.merged = tf.summary.merge_all()

  def get_random_choice(self, data, num):
    len_all = len(data)
    shuf = np.random.permutation(np.arange(len_all))[0 : num]
    res = [data[i] for i in shuf]
    return res

  def get_data_with_mode(self, data, num=None, mode=["random", 0], upd=None):
    len_all = len(data)
    if not num:
      num = len_all
    slt = []
    if mode[0] == "random":
      slt = self.get_random_choice(data, num)
    elif mode[0] == "in_order":
      beg = (num * mode[1]) % len_all
      end = (num * (mode[1] + 1)) % len_all
      if end <= beg:
        slt.extend(data[beg : len_all])
        slt.extend(data[0 : end])
      else:
        slt = data[beg : end]
    elif mode[0] == "last":
      slt = data[len_all : len_all - num - 1 : -1]
    elif mode[0] == "lifo":
      lifo_pro = 0.3
      if np.random.uniform() < lifo_pro:
        if not upd:
          slt = self.get_random_choice(data, num)
        else:
          len_upd = len(upd)
          if len_upd < num:
            slt = upd
            slt.extend(self.get_random_choice(data, num - len_upd))
          else:
            slt = self.get_random_choice(upd, num)
      else:
        slt = self.get_random_choice(data, num)
    return slt

  def get_batch_from_lib(self, data_type="train", mode=["random", 0]):
    upd = None
    if data_type == "train":
      mdata, lables = self.load_data(self.data_bcp_path)
    if data_type == "dev":
      mdata, lables = self.load_data(self.data_bcp_path)
      cur = datetime.date.today()
      while True:
        cur = cur - datetime.timedelta(days=1)
        fn = "data_out_%s.txt" % cur.strftime("%y%m%d")
        if os.path.exists(fn):
          break
      ud, ul = self.load_data(fn)
      upd = zip(ud, ul)
    if data_type == "test":
      mdata, lables = self.load_data(self.shouzhu_path)
    slt = self.get_data_with_mode(zip(mdata, lables), num=self.batch_size, mode=mode, upd=upd)
    rdata, rlables = [], []
    for x, y in slt:
      rdata.append(x)
      if 0 < len(y):
        rlables.append(y[self.train_type])
      else:
        rlables.append(0)
    return rdata, rlables

  def get_ckpt_dir(self):
    out_dir = os.path.join(cur_dir, "cnn_output/model_%s/" % self.train_type + time.strftime("%y%m%d", time.localtime()))
    ckpt_dir = out_dir + '/model.ckpt'
    if not os.path.exists(out_dir):
      os.makedirs(ckpt_dir)
    return ckpt_dir

  def train(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=self.graph, config=config) as sess:
      data_writer = tf.summary.FileWriter("logs/dqn", sess.graph)
      sess.run(tf.global_variables_initializer())
      time_loop = 20
      time_rwd = 20
      time_test = 100
      time_fre = 200
      time_final = 1000000
      loss, acc = 0.0, 0.0
      lr_init = 1e-4
      lr_end = 1e-5
      i = 0
      train_times = 2000
      while True:
        if i < train_times * 1 / 4:
          cur_lr = lr_init
        else:
          cur_lr = lr_end
        sess.run(self._lr_update, feed_dict={self._new_lr: cur_lr})
        x_batch, y_batch = self.get_batch_from_lib(data_type="train", mode=["in_order", i])
        for j in range(time_loop):
          q_tar = sess.run(self.next_logits, feed_dict={self.input_x: x_batch, self.prob_holder: self.keep_probs[self.train_type]})
          tar_lable = sess.run(self.tar_prediction, feed_dict={self.input_x: x_batch, self.prob_holder: self.keep_probs[self.train_type], self.target_logits: q_tar})
          summary, _, loss, acc = sess.run([self.merged, self.train_step, self.loss, self.accuracy], feed_dict={self.input_x: x_batch, self.input_y: tar_lable, self.prob_holder: self.keep_probs[self.train_type], self.target_logits: q_tar})
          data_writer.add_summary(summary, i * time_loop + j)
        cur_step = (i + 1) * time_loop
        if cur_step % time_rwd == 0:
          if cur_step % time_test == 0:
            print "train type:", self.train_type, "pid:", os.getpid()
            print "current step:", cur_step, ", learning rate:", cur_lr
            print "train loss: %g, accuracy: %g" % (loss, acc)
          x_batch, y_batch = self.get_batch_from_lib(data_type="dev", mode=["lifo", 0])
          q_tar = sess.run(self.next_logits, feed_dict={self.input_x: x_batch, self.prob_holder: self.keep_probs[self.train_type]})
          e_tar = sess.run(self.prediction, feed_dict={self.input_x: x_batch, self.prob_holder: self.keep_probs[self.train_type]})
          f_diff = sess.run(self.fin_diff, feed_dict={self.input_x: x_batch, self.prob_holder: self.keep_probs[self.train_type], self.input_y: y_batch, self.target_logits: q_tar})
          print "e_tar:", e_tar
          #print "f_diff:", f_diff
          tr_loss, tr_acc, prob, _ = sess.run([self.loss, self.accuracy, self.probability, self.train_step], feed_dict={self.input_x: x_batch, self.input_y: y_batch, self.prob_holder: self.keep_probs[self.train_type], self.target_logits: q_tar})
          if cur_step % time_test == 0:
            print "devir loss: %g, accuracy: %g" % (tr_loss, tr_acc)
          sess.run(self.replace_target_op)
          take_down_pro = 0.8
          take_down_step_pro = 0.6
          if self.his_acc[self.train_type] <= tr_acc and (take_down_pro < acc or take_down_step_pro * train_times * time_loop < cur_step):
            self.ckpt_dir = self.get_ckpt_dir()
            self.saver.save(sess, self.ckpt_dir)
            self.his_acc[self.train_type] = tr_acc.item()
          if self.his_acc[self.train_type] <= tr_acc:
            self.his_acc = self.load_history_accuracy()
            self.his_acc[self.train_type] = tr_acc.item()
            with open(self.his_path, "w") as of:
              json.dump(self.his_acc, of)
        i += 1
        if i == time_final:
          i == 0

def main_proc(num=0, use_gpus="0"):
  lc = LotteryCNN(train_type=num, use_gpus=use_gpus)
  lc.train()

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    main_proc(0, "6")
  elif len(sys.argv) == 2:
    num = int(sys.argv[1])
    main_proc(num, "6")
  else:
    num = int(sys.argv[1])
    use_gpus = sys.argv[2]
    main_proc(num, use_gpus)
  #p = Pool()
  #for i in range(2):
  #  gpus = "%s" % (i)
  #  p.apply_async(main_proc, args=(i, gpus, ))
  #p.close()
  #p.join()
