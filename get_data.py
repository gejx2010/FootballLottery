#!/usr/bin/dev python
#coding=utf-8
#Authou: spgoal
#Date: 2018-02-04

import requests
import json
import chardet
import time
import datetime
import numpy
import os
import shutil
from lxml import etree
from web_model import *

class GetLotteryData(object):
  def __init__(self):
    self.cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(self.cur_dir)
    self.model_name = "www.lottery.gov.cn"
    self.web_model = WebModel(self.model_name)
    self.html_file_name = "lottery_file.html"
    self.parser = etree.HTMLParser()
    self.res_list = []
    self.shouzhu_match_list = []
    self.shouzhu_data_list = []
    # file path
    self.ft_file = os.path.join(self.cur_dir, "data/football_team.txt")
    self.ml_file = os.path.join(self.cur_dir, "data/match_league.txt")
    self.his_match_file = os.path.join(self.cur_dir, "data/history_match_id.txt")
    self.total_fb_file = os.path.join(self.cur_dir, "data/total_match_result.txt")
    self.backup_fb_file = os.path.join(self.cur_dir, "data/backup_match_result.txt")
    self.cur_fb_file = os.path.join(self.cur_dir, "data/match_out_%s.txt" % time.strftime("%y%m%d", time.localtime()))
    self.total_data_file = os.path.join(self.cur_dir, "data/total_data.txt")
    self.backup_data_file = os.path.join(self.cur_dir, "data/backup_data.txt")
    self.cur_data_file = os.path.join(self.cur_dir, "data/data_out_%s.txt" % time.strftime("%y%m%d", time.localtime()))
    self.ftn = self.load_file(self.ft_file)
    self.mln = self.load_file(self.ml_file)
    self.hmn = self.load_file(self.his_match_file)
    self.shouzhu_match_file = os.path.join(self.cur_dir, "data/shouzhu_match.txt")
    self.shouzhu_data_file = os.path.join(self.cur_dir, "data/shouzhu_data.txt")
    # constant for date length
    self.len_info = 16
    self.len_prob = 16
    self.len_prob_rang = 16
    self.len_prob_goals = 8
    self.len_prob_merge = 8
    self.len_prob_points = 8
    self.type_dict = {
                         "goals": MAX_GOALS + 2,
                         "merge": len(merge_match_res_dict) + 1,
                         "points": len(point_res_dict) + 1
                      }

  def write_dict_to_file(self, outfile, res_dict):
    f = open(outfile, "w")
    rev = dict(zip(res_dict.values(), res_dict.keys()))
    kl = sorted(rev.keys(), key=lambda x: int(x))
    for k in kl:
      f.write(rev[k].encode("utf-8") + "\1" + str(k).encode("utf-8") + "\n")
    f.close()

  def merge_output_file(self, res_list, match_file):
    if not os.path.exists(match_file):
      crt = open(infile, "w")
      crt.close()
    match_list = []
    if 0 < os.path.getsize(match_file):
      rf = open(match_file, "r")
      match_list = json.load(rf)
      rf.close()
    for dd in res_list:
      if dd["result"]:
        match_id = self.get_match_dict_id(dd)
        if not match_id in self.hmn:
          self.hmn[match_id] = len(self.hmn)
          match_list.append(dd)
    of = open(match_file, "w")
    of.write(json.dumps(match_list, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))
    of.close()
        
  def write_data(self):
    shutil.copy(self.total_fb_file, self.backup_fb_file)
    shutil.copy(self.total_data_file, self.backup_data_file)
    self.merge_output_file(self.res_list, self.total_fb_file)
    gld.turn_file_to_data_file(gld.total_fb_file, gld.total_data_file)
    self.write_dict_to_file(self.ft_file, self.ftn)
    self.write_dict_to_file(self.ml_file, self.mln)
    self.write_dict_to_file(self.his_match_file, self.hmn)
  
  def load_file(self, infile):
    res = {}
    f = open(infile, "r")
    for ll in f:
      l_seg = ll.strip().split("\1")
      name, num = l_seg[0].decode("utf-8"), l_seg[1].decode("utf-8")
      res[name] = num
    f.close()
    return res

  def write_data_file(self, url=""):
    if not url:
      resp = requests.get(self.url)
      f = open(self.html_file_name, "w")
      f.write(resp.text.encode("utf-8"))
      f.close()

  def get_data(self, outfile=""):
    if not outfile:
      outfile = self.cur_fb_file
    of = open(outfile, "w")
    self.res_list = self.web_model.get_data()
    of.write(json.dumps(self.res_list, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))
    self.write_list_to_data_file(self.res_list)
    of.close()
    return self.res_list

  '''
  function: get shou zhu match, write it into shouzhu_match_list and shouzhu_data_list
  '''
  def get_shouzhu_match(self):
    self.shouzhu_match_list = self.web_model.get_shouzhu_match()
    of = open(self.shouzhu_match_file, "w")
    of.write(json.dumps(self.shouzhu_match_list, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))
    self.shouzhu_data_list = self.turn_list_to_data(self.shouzhu_match_list)
    self.write_list_to_data_file(self.shouzhu_match_list, outfile=self.shouzhu_data_file)
    of.close()
    return self.shouzhu_match_list

  def turn_match_time_to_data(self, match_time):
    time_seg = match_time.split()
    date, time = time_seg[1], time_seg[2]
    date_seg = date.split("-")
    year, month, day = date_seg[0], date_seg[1], date_seg[2]
    time_sub_seg = time.split(":")
    hour, minute = time_sub_seg[0], time_sub_seg[1]
    return year, month, day, hour, minute

  def turn_dict_info_to_data(self, match_dict, match_data):
    info = match_dict["info"]
    cur_data = []
    name_div_num = 1000
    cur_data = self.turn_name_to_data(info["home"], cur_data, self.ftn, name_div_num)
    cur_data = self.turn_name_to_data(info["guest"], cur_data, self.ftn, name_div_num)
    # liga
    week = info["liga"][0:2]
    week_div_num = 10
    cur_data.append(float(self.turn_week_to_num(week)) / week_div_num)
    rank = info["liga"][2:5]
    rank_div_num = 100
    cur_data.append(float(rank) / rank_div_num)
    liga = info["liga"][6:]
    liga_div_num = 100
    cur_data = self.turn_name_to_data(liga, cur_data, self.mln, liga_div_num)
    # match_time
    year, month, day, hour, minute = self.turn_match_time_to_data(info["match_time"])
    year_pro, month_pro, day_pro, hour_pro, minute_pro = 10000, 100, 100, 100, 100
    cur_data.append(float(year) / year_pro)
    cur_data.append(float(month) / month_pro)
    cur_data.append(float(day) / day_pro)
    cur_data.append(float(hour) / hour_pro)
    cur_data.append(float(minute) / minute_pro)
    while len(cur_data) < self.len_info:
      cur_data.append(0)
    match_data.extend(cur_data)
    return match_data

  def turn_name_to_data(self, name, match_data, name_dict, div_num=1):
    if name in name_dict:
      match_data.append(float(name_dict[name]) / div_num)
    else:
      name_dict[name] = len(name_dict)
      match_data.append(float(len(name_dict)) / div_num)
    return match_data

  def turn_week_to_num(self, week):
    week_str = ["一", "二", "三", "四", "五", "六", "日"]
    for _, s in enumerate(week_str):
      if s in week.encode("utf-8"):
        return _ + 1

  def turn_dict_prob_to_data(self, match_dict, match_data):
    prob_list = match_dict["probability"][::-1]
    for i in range(self.len_prob):
      if i < len(prob_list):
        prob = prob_list[i]
        turn, win, draw, lose = int(prob["turn"]), float(prob["win"]), float(prob["draw"]), float(prob["lose"])
      else:
        turn, win, draw, lose = 0, 0.0, 0.0, 0.0
      match_data.extend([turn, win, draw, lose])
    return match_data
      
  def turn_dict_prob_rang_to_data(self, match_dict, match_data):
    prob_list = match_dict["probability_rang_qiu"][::-1]
    for i in range(self.len_prob):
      if i < len(prob_list):
        prob = prob_list[i]
        turn, rang_qiu, win, draw, lose = int(prob["turn"]), int(prob["rang_qiu_shu"]), float(prob["win"]), float(prob["draw"]), float(prob["lose"])
      else:
        turn, rang_qiu, win, draw, lose = 0, 0, 0.0, 0.0, 0.0
      match_data.extend([turn, rang_qiu, win, draw, lose])
    return match_data
      
  def turn_dict_prob_goals_to_data(self, match_dict, match_data):
    prob_list = match_dict["probability_goals"][::-1]
    plen = self.type_dict["goals"]
    for i in range(self.len_prob_goals):
      if i < len(prob_list):
        prob = prob_list[i]
        plen = len(prob)
        turn = prob["turn"]
        match_data.append(int(turn))
        # goals < 10
        last = 0
        for j in range(10):
          m = prob.get("%d" % j, "") 
          if not m:
            last = j
            break
          match_data.append(float(m))
        m = prob.get("%d+" % last, "")
        if m:
          match_data.append(float(m))
      else:
        match_data.append(0)
        for i in range(plen - 1):
          match_data.append(0.0)
      match_data.append(0.0)
    return match_data
      
  def turn_dict_prob_merge_to_data(self, match_dict, match_data):
    prob_list = match_dict["probability_merge"][::-1]
    key_list = [u"胜胜", u"胜平", u"胜负", u"平胜", u"平平", u"平负", u"负胜", u"负平", u"负负"]
    plen = self.type_dict["merge"]
    for i in range(self.len_prob_merge):
      if i < len(prob_list):
        prob = prob_list[i]
        plen = len(prob)
        turn = prob["turn"]
        match_data.append(int(turn))
        for key in key_list:
          match_data.append(float(prob[key]))
      else:
        match_data.append(0)
        for i in range(plen - 1):
          match_data.append(0.0)
    return match_data
  
  def turn_match_turn_to_num(self, s):
    num_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for num in num_list:
      if num in s:
        return int(num)
    return 0
      
  def turn_dict_prob_points_to_data(self, match_dict, match_data):
    prob_list = match_dict["probability_points"][::-1]
    plen = self.type_dict["points"]
    key_list = sorted(point_res_dict.keys())
    for i in range(self.len_prob_points):
      if i < len(prob_list):
        prob = prob_list[i]
        match_data.append(self.turn_match_turn_to_num(prob["turn"]))
        for k in key_list:
          if k != "turn":
            match_data.append(float(prob.get(k, 0.0)))
      else:
        match_data.append(0)
        for i in range(plen - 1):
          match_data.append(0.0)
    return match_data
      
  def turn_dict_result_to_data(self, match_dict):
    mr = match_dict["result"]
    if not mr:
      return []
    res = []
    if not mr["match_result"]:
      res.append(0)
    else:
      result_num = match_res_dict[mr["match_result"].encode("utf-8")]
      res.append(result_num)
    if not mr["match_rang_result"]:
      res.append(0)
    else:
      result_rang_num = match_res_dict[mr["match_rang_result"].encode("utf-8")]
      res.append(result_rang_num)
    if not mr["match_merge_result"]:
      res.append(0)
    else:
      result_merge_num = merge_match_res_dict[mr["match_merge_result"].encode("utf-8")]
      res.append(result_merge_num)
    if not mr["match_goals"]:
      res.append(0)
    elif mr["match_goals"] != "7+":
      goals = int(mr["match_goals"])
      res.append(goals)
    else:
      goals = 7
      res.append(goals)
    if not mr["match_point"]:
      res.append(0)
    else:
      result_point_num = point_res_dict[mr["match_point"].encode("utf-8")]
      res.append(result_point_num)
    return res
      
  def turn_dict_to_data(self, match_dict):
    # decode info
    match_data = []
    match_data = self.turn_dict_info_to_data(match_dict, match_data)
    ss = self.len_info
    if len(match_data) < ss:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode probability
    match_data = self.turn_dict_prob_to_data(match_dict, match_data)
    ss += self.len_prob * 4
    if len(match_data) < ss:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode probability rang qiu
    match_data = self.turn_dict_prob_rang_to_data(match_dict, match_data)
    ss += self.len_prob * 5
    if len(match_data) < ss:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode probability goals
    match_data = self.turn_dict_prob_goals_to_data(match_dict, match_data)
    ss += self.len_prob_goals * 10
    if len(match_data) < ss:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode probability merge
    match_data = self.turn_dict_prob_merge_to_data(match_dict, match_data)
    ss += self.len_prob_merge * 10
    if len(match_data) < ss:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode probability points
    match_data = self.turn_dict_prob_points_to_data(match_dict, match_data)
    if len(match_data) < 576:
      print "len of match data:", len(match_data)
      for x, v in match_dict["info"].items():
        print x, v
    # decode match result
    res = self.turn_dict_result_to_data(match_dict)
    return (match_data, res)

  def turn_list_to_data(self, res_list):
    out_list = []
    for dd in res_list:
      if not "info" in dd:
        continue
      if not dd["info"]:
        continue
      out_list.append(self.turn_dict_to_data(dd))
    return out_list
   
  def turn_file_to_data(self, infile):
    if os.path.getsize(infile) < 1:
      return []
    rf = open(infile, "r")
    res_list = json.load(rf)
    rf.close()
    return self.turn_list_to_data(res_list)
   
  def write_list_to_data_file(self, res_list, outfile=""):
    if not outfile:
      outfile = self.cur_data_file
    of = open(outfile, "w")
    data_list = self.turn_list_to_data(res_list)
    of.write(json.dumps(data_list, ensure_ascii=False, indent=4, sort_keys=True))
    of.close()

  def turn_file_to_data_file(self, infile, outfile):
    data_list = self.turn_file_to_data(infile)
    of = open(outfile, "w")
    of.write(json.dumps(data_list, ensure_ascii=False, indent=4, sort_keys=True))
    of.close()

  def get_match_dict_id(self, match_dict):
    if not "info" in match_dict:
      return ""
    if not match_dict["info"]:
      return ""
    info = match_dict["info"]
    return info["match_time"] + " " + info["liga"] + " " + info["home"] + " vs " + info["guest"]

  def is_same_match(self, match_a, match_b):
    return match_a["info"] == match_b["info"]

  def sort_total_file(self, json_file_name, backup_file_name):
    shutil.copy(json_file_name, backup_file_name)
    rf = open(json_file_name, "r")
    json_list = json.load(rf)
    rf.close()
    func = lambda x, y: x if y in x else x + [y]
    new_json_list = reduce(func, [[], ] + json_list)
    wf = open(json_file_name, "w")
    wf.write(json.dumps(new_json_list, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))
    wf.close()

  # example: football_team.txt
  def move_team_name_space(self, infile):
    rf = open(infile, "r")
    old_lines = rf.readlines()
    rf.close()
    of = open(infile, "w")
    for ll in old_lines:
      l_seg = ll.strip().split("\1")
      name, _ = l_seg[0].split(" ")[0], l_seg[1]
      of.write(name + "\1" + _ + "\n")

  # example: total_match_result.txt
  def move_team_name_space_in_json_file(self, infile):
    rf = open(infile, "r")
    inlist = json.load(rf)
    rf.close()
    of = open(infile, "w")
    for ll in inlist:
      home = ll["info"]["home"].split(" ")[0]
      guest = ll["info"]["guest"].split(" ")[0]
      ll["info"]["home"] = home
      ll["info"]["guest"] = guest
    of.write(json.dumps(inlist, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))

  def renew_match_id_file(self):
    rf = open(self.total_fb_file, "r")
    inlist = json.load(rf)
    rf.close()
    of = open(self.his_match_file, "w")
    for _, ll in enumerate(inlist):
      match_id = self.get_match_dict_id(ll)
      of.write(match_id.encode("utf-8") + "\1" + str(_).encode("utf-8") + "\n")
    of.close()

  def load_shouzhu_match(self):
    rf = open(self.shouzhu_match_file, "r")
    self.shouzhu_match_list = json.load(rf)
    rf.close()
    rfd = open(self.shouzhu_data_file, "r")
    self.shouzhu_data_list = json.load(rfd)
    rfd.close()

  def unique_match_list(self):
    with open(self.total_fb_file, "r") as rf:
      match_list = json.load(rf)
      func = lambda x, y: x if y in x else x + [y]
      new_list = reduce(func, [[], ] + match_list)
    with open(self.total_fb_file, "w") as of:
      of.write(json.dumps(new_list, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8"))

  def redict(self, ori_dict, ofn):
    new_dict = {}
    for k, v in ori_dict.items():
      v = int(v)
      if v in new_dict:
        new_dict[v + 1] = k
      else:
        new_dict[v] = k
    res_dict = dict(zip(new_dict.values(), new_dict.keys()))
    self.write_dict_to_file(ofn, res_dict)

if __name__ == "__main__":
  gld = GetLotteryData()
  gld.get_data()
  gld.write_data()
  gld.get_shouzhu_match()
  #gld.unique_match_list()
  #gld.turn_file_to_data_file(gld.total_fb_file, gld.total_data_file)
  #gld.redict(gld.ftn, gld.ft_file)
  #gld.sort_total_file(gld.total_fb_file, gld.backup_fb_file)
  #gld.sort_total_file(gld.total_data_file, gld.backup_data_file)
  #gld.move_team_name_space(gld.ft_file)
  #gld.move_team_name_space_in_json_file(gld.total_fb_file)
  #gld.renew_match_id_file()
