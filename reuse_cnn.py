#!/usr/bin/dev python
#coding=utf-8
#author: spgoal
#time: 2018-02-11

import os
import sys
import numpy as np
import tensorflow as tf
import time
import json
from get_data import *
from train_data import *
import train_data_cnn as tdc

cur_dir = os.path.dirname(os.path.abspath(__file__))
cnn_dir = os.path.join(cur_dir, "cnn_output")
only_cnn_dir = os.path.join(cur_dir, "only_cnn_output")
gld = GetLotteryData()
gld.load_shouzhu_match()
rf = open("match_out_180212.txt", "r")
match_dict = json.load(rf)
rf.close()

def find_latest_file(dir_name, res_type):
  res, out = 0, ""
  for ff in os.listdir(dir_name):
    if os.path.isfile(ff):
      seg = ff.split("graph_")[0].split("_type")[0]
      if res < int(data) and ("type_" + str(res_type)) in ff:
        res = int(data)
        out = ff
  return out

def turn_query_to_dict(query):
  for ll in match_dict:
    info = ll[u"info"]
    print "info home:", info[u"home"]
    print "is equal:", info[u"home"].encode("utf-8") == query["home"]
    if info[u"home"].encode("utf-8") == query["home"] and info[u"guest"].encode("utf-8") == query["guest"]:
      date = query.get("date", "")
      if date and date in info[u"match_time"].encode("utf-8"):
        return ll
  return None

def fullfill(query_data):
  np_data = np.reshape(query_data, [DATA_HEIGHT, DATA_WIDTH, 1])
  zero_data = np.zeros([DATA_HEIGHT, DATA_HEIGHT - DATA_WIDTH, 1])
  np_data = np.concatenate([np_data, zero_data], axis = 1)
  return np_data

def get_name_by_rank(d, rank):
  res = []
  new_d = dict(zip(d.values(), d.keys()))
  for r in rank:
    res.append(new_d[r])
  return res

def translate(res, res_type):
  if res_type <= 1:
    return get_name_by_rank(match_res_dict, res)
  elif res_type == 2:
    return get_name_by_rank(merge_match_res_dict, res)
  elif res_type == 4:
    return get_name_by_rank(point_res_dict, res)
  else:
    return res

def reuse_cnn(ckpt_path="", res_type=0, input_query=None, input_file=None, output_file=None):
  with tf.Session() as sess:
    # restore saver
    ckpt_file = tf.train.latest_checkpoint(ckpt_path)
    saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_file))
    saver.restore(sess, ckpt_file)
    graph = tf.get_default_graph()
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    input_y = graph.get_operation_by_name("input_y").outputs[0]
    keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
    loss = graph.get_operation_by_name("loss/loss").outputs[0]
    train_step = graph.get_operation_by_name("loss/loss").outputs
    prediction = graph.get_operation_by_name("output/prediction").outputs[0]
    probability = graph.get_operation_by_name("output/probability").outputs[0]
    accuracy = graph.get_operation_by_name("output/accuracy").outputs[0]
    if input_query:
      query_dict = turn_query_to_dict(input_query)
      query_data, _ = gld.turn_dict_to_data(query_dict)
      x_test = []
      x_test.append(fullfill(query_data))
      while len(x_test) < BATCH_SIZE:
        x_test.append(np.zeros([DATA_HEIGHT, DATA_HEIGHT, 1]))
      y_test = []
      while len(y_test) < BATCH_SIZE: 
        y_test.append(0)
      print "res_type:", res_type
      loss, _, pre, prob, acc = sess.run([loss, train_step, prediction, probability, accuracy], feed_dict={input_x: x_test, input_y: y_test, keep_prob: KEEP_PROBS[res_type]})
      trans_pre = translate(pre[0], res_type)
      print "prediction:", trans_pre
      print json.dumps(trans_pre, ensure_ascii=False)
      print "probability:", prob[0]
      return (pre[0], trans_pre, prob[0])

'''
function: turn data to input format for tensor input
'''
def turn_data_to_input_format(input_data, div=False):
  data_len = len(input_data)
  np_data = np.array(input_data)
  np_data = np.reshape(np_data, [-1, DATA_HEIGHT, DATA_WIDTH, 1])
  zero_pad = np.zeros([data_len, DATA_HEIGHT, DATA_HEIGHT - DATA_WIDTH, 1])
  np_data = np.concatenate([np_data, zero_pad], axis=2)
  if div:
    np_data = np_data.__truediv__(1000)
  return np_data

def reuse_cnn_for_subpath(ckpt_path="", res_type=0, drop_prob=0.0):
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      # restore saver
      ckpt_file = tf.train.latest_checkpoint(ckpt_path)
      saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_file))
      saver.restore(sess, ckpt_file)
      input_x = graph.get_operation_by_name("input_x").outputs[0]
      input_y = graph.get_operation_by_name("input_y").outputs[0]
      keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
      loss = graph.get_operation_by_name("loss/loss").outputs[0]
      train_step = graph.get_operation_by_name("loss/loss").outputs
      prediction = graph.get_operation_by_name("output/prediction").outputs[0]
      probability = graph.get_operation_by_name("output/probability").outputs[0]
      accuracy = graph.get_operation_by_name("output/accuracy").outputs[0]
      rank = 0
      res = {}
      res["prediction"] = []
      res["translation"] = []
      res["probability"] = []
      while rank * tdc.BATCH_SIZE < len(gld.shouzhu_data_list):
        beg = rank * tdc.BATCH_SIZE
        end = (rank + 1) * tdc.BATCH_SIZE
        end = end if end < len(gld.shouzhu_data_list) else len(gld.shouzhu_data_list)
        x_test = [x[0] for x in gld.shouzhu_data_list[beg : end]]
        x_test = turn_data_to_input_format(x_test, div=False)
        if len(x_test) < tdc.BATCH_SIZE:
          zero_pad = np.zeros([tdc.BATCH_SIZE - len(x_test), tdc.DATA_HEIGHT, tdc.DATA_HEIGHT, 1])
          x_test = np.concatenate([x_test, zero_pad], axis=0)
        y_test = []
        while len(y_test) < tdc.BATCH_SIZE: 
          y_test.append(0)
        print "res_type:", res_type
        if not drop_prob:
          drop_prob = tdc.KEEP_PROBS[res_type]
        lss, pre, prob, acc = sess.run([loss, prediction, probability, accuracy], feed_dict={input_x: x_test, input_y: y_test, keep_prob: 1.0})
        trans_pre = translate(pre, res_type)
        print "loss:", lss
        print "accuracy", acc
        print "prediction:", pre
        for tp in trans_pre:
          print "transloation prediction:", tp
        print "probability:", prob
        res["prediction"].extend(pre)
        res["translation"].extend(trans_pre)
        res["probability"].extend(prob)
        rank += 1
      return res

def reuse_dqn_for_subpath(ckpt_path="", res_type=0, drop_prob=0.0):
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      # restore saver
      ckpt_file = tf.train.latest_checkpoint(ckpt_path)
      saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_file))
      saver.restore(sess, ckpt_file)
      input_x = graph.get_operation_by_name("input_x").outputs[0]
      input_y = graph.get_operation_by_name("input_y").outputs[0]
      keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
      eval_logits = graph.get_operation_by_name("eval_net/full_connect_3/h_fc3").outputs[0]
      next_logits = graph.get_operation_by_name("target_net/full_connect_3/h_fc3").outputs[0]
      target_logits = graph.get_operation_by_name("target_logits").outputs[0]
      #loss = graph.get_operation_by_name("loss/loss").outputs[0]
      prediction = graph.get_operation_by_name("orig_output/orig_prediction").outputs[0]
      probability = graph.get_operation_by_name("orig_output/orig_probability").outputs[0]
      accuracy = graph.get_operation_by_name("orig_output/orig_accuracy").outputs[0]
      rank = 0
      res = {}
      res["prediction"] = []
      res["translation"] = []
      res["probability"] = []
      while rank * BATCH_SIZE < len(gld.shouzhu_data_list):
        beg = rank * BATCH_SIZE
        end = (rank + 1) * BATCH_SIZE
        end = end if end < len(gld.shouzhu_data_list) else len(gld.shouzhu_data_list)
        x_test = [x[0] for x in gld.shouzhu_data_list[beg : end]]
        x_test = turn_data_to_input_format(x_test, div=True)
        if len(x_test) < BATCH_SIZE:
          zero_pad = np.zeros([BATCH_SIZE - len(x_test), DATA_HEIGHT, DATA_HEIGHT, 1])
          x_test = np.concatenate([x_test, zero_pad], axis=0)
        y_test = []
        while len(y_test) < BATCH_SIZE: 
          y_test.append(0)
        print "res_type:", res_type
        if not drop_prob:
          drop_prob = KEEP_PROBS[res_type]
        pre, prob = sess.run([prediction, probability], feed_dict={input_x: x_test, keep_prob: drop_prob})
        trans_pre = translate(pre, res_type)
        print "prediction:", pre
        for tp in trans_pre:
          print "transloation prediction:", tp
        print "probability:", prob
        res["prediction"].extend(pre)
        res["translation"].extend(trans_pre)
        res["probability"].extend(prob)
        rank += 1
      return res

def merge_result(res):
  fin = {}
  match_type_dict = {0: "胜平负", 
                     1: "让球胜平负",
                     2: "半场胜平负",
                     3: "进球数",
                     4: "比分",
                    }
  for i, tp in enumerate(res):
    for j, mt in enumerate(tp):
      mname = mt["name"].encode("utf-8")
      if not mname in fin:
        fin[mname] = []
      dd = {}
      dd["type"] = match_type_dict[i]
      dd["prediction"] = mt["prediction"]
      dd["translation"] = mt["translation"] if not (isinstance(mt["translation"], int) or isinstance(mt["translation"], numpy.int64)) else str(mt["translation"].item())
      dd["probability"] = mt["probability"]
      fin[mname].append(dd)
  return fin

def reuse_cnn_for_all(date="", out_type=[], drop_prob=0.0, struct="cnn"):
  final_res = []
  if not out_type:
    out_type = range(OUTPUT_TYPE)
  for i in out_type:
    if not date:
      cur_date = get_newest_date(i, struct=struct)
      print "use training data at date:", cur_date
    else:
      cur_date = date
    sub_res = []
    if struct == "cnn":
      ckpt_path = os.path.join(only_cnn_dir, "model_%s/%s" % (i, cur_date))
      if not os.path.exists(ckpt_path):
        continue
      print "res_type:", i, "ckpt_path:", ckpt_path
      pre = reuse_cnn_for_subpath(ckpt_path, res_type=i, drop_prob=drop_prob)
    elif struct == "dqn":
      ckpt_path = os.path.join(cnn_dir, "model_%s/%s" % (i, cur_date))
      if not os.path.exists(ckpt_path):
        continue
      print "res_type:", i, "ckpt_path:", ckpt_path
      pre = reuse_dqn_for_subpath(ckpt_path, res_type=i, drop_prob=drop_prob)
    for i, match in enumerate(gld.shouzhu_match_list):
      dd = {}
      dd["name"] = gld.get_match_dict_id(match)
      dd["prediction"] = pre["prediction"][i]
      dd["translation"] = pre["translation"][i]
      dd["probability"] = pre["probability"][i].tolist()
      sub_res.append(dd)
    final_res.append(sub_res)
  final_res = merge_result(final_res)
  return final_res

def get_newest_date(out_type, struct="dqn"):
  res = 0
  cur_dir = cnn_dir if struct == "dqn" else only_cnn_dir
  for ff in os.listdir(cur_dir + "/model_%d" % out_type):
    if os.path.isdir(cur_dir + "/model_%d/" % out_type + ff):
      meta_file = cur_dir + "/model_%d/" % out_type + ff + "/model.ckpt.meta"
      ckpt_file = cur_dir + "/model_%d/" % out_type + ff + "/checkpoint"
      if os.path.exists(meta_file) and os.path.exists(ckpt_file):
        dd = int(ff)
        if res < dd:
          res = dd
  return str(res)

if __name__ == "__main__":
  res = reuse_cnn_for_all(date="180228")
  print "res:", res
  with open("test.txt", "w") as rf:
    rf.write(json.dumps(res, indent=4, ensure_ascii=False))
