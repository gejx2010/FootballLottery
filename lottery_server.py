#!/usr/bin/dev python
#coding=utf-8
#authou: spgoal
#date: Feb 23th, 2018

import os
import sys
import time
import json
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from reuse_cnn import *
#from prepared_server_result import *
import requests
from tornado.options import define, options

# scheduler
from datetime import datetime

define("port", default=8686, help="run on the given port", type=int)

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

class IndexHandler(tornado.web.RequestHandler):
  def initialize(self):
    self.res_file = "cur_result_for_server_%s.txt"
    self.res_date_file = "cur_result_for_server_%s_%s.txt"
 
  def filter_key(self, res_key):
    if self.key:
      key_params = self.key.split("|")
      t = 0
      for k in key_params:
        if not k in res_key: 
          break
        t += 1
      if t == len(key_params):
        return True
    else:
      return True
    return False

  def filter_mgt(self, res_key):
    pro = self.fin[res_key][0]["probability"]
    for p in pro:
      if self.mgt < p:
        return True
    return False
      
  def filter_rgt(self, res_key):
    pro = self.fin[res_key][1]["probability"]
    for p in pro:
      if self.rgt < p:
        return True
    return False
      
  def filter_hgt(self, res_key):
    pro = self.fin[res_key][2]["probability"]
    for p in pro:
      if self.hgt < p:
        return True
    return False
      
  def filter_ggt(self, res_key):
    pro = self.fin[res_key][3]["probability"]
    for p in pro:
      if self.ggt < p:
        return True
    return False
      
  def filter_pgt(self, res_key):
    pro = self.fin[res_key][4]["probability"]
    for p in pro:
      if self.pgt < p:
        return True
    return False
      
  def make_dict(self, key_list):
    dd = {}
    for k in key_list:
      dd[k] = self.res[k]
    return dd

  def get(self):
    self.date = self.get_argument("date", "")
    self.stt = self.get_argument("stt", "")
    if not self.stt:
      self.stt = "dqn"
    if self.date:
        with open(self.res_date_file % (self.date, self.stt), "r") as rf:
          self.res = json.load(rf)
    else:
      with open(self.res_file % self.stt, "r") as rf:
        self.res = json.load(rf)
    self.key = self.get_argument("key", "")
    self.mgt = float(self.get_argument("mgt", "0.0"))
    self.rgt = float(self.get_argument("rgt", "0.0"))
    self.hgt = float(self.get_argument("hgt", "0.0"))
    self.ggt = float(self.get_argument("ggt", "0.0"))
    self.pgt = float(self.get_argument("pgt", "0.0"))
    self.fin = self.res
    self.fin = self.make_dict(filter(self.filter_key, self.fin))
    self.fin = self.make_dict(filter(self.filter_mgt, self.fin))
    self.fin = self.make_dict(filter(self.filter_rgt, self.fin))
    self.fin = self.make_dict(filter(self.filter_hgt, self.fin))
    self.fin = self.make_dict(filter(self.filter_ggt, self.fin))
    self.fin = self.make_dict(filter(self.filter_pgt, self.fin))
    self.to_simple = self.get_argument("sim", "")
    if not self.to_simple:
      self.write(json.dumps(self.fin, ensure_ascii=False, indent=4))
    else:
      self.sim_fin = {}
      for kk, vv in self.fin.items():
        ndd = []
        ndd.append((vv[0]["translation"], max(vv[0]["probability"])))
        ndd.append((vv[1]["translation"], max(vv[1]["probability"])))
        self.sim_fin[kk] = ndd
      self.write(json.dumps(self.sim_fin, ensure_ascii=False, indent=4))

def main():
  tornado.options.parse_command_line()
  app = tornado.web.Application(handlers=[(r'/', IndexHandler),])
  http_server = tornado.httpserver.HTTPServer(app)
  http_server.listen(options.port)
  tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
  main()
