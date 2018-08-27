#!/usr/bin/dev python
#coding=utf-8
#Author: spgoal
#date: Apr. 18th, 2018

import os
import sys
import json
from lxml import etree

# dict for chinese word
match_res_dict = {"胜": 0, "平": 1, "负": 2}
merge_match_res_dict = {
                        "胜胜": 0,
                        "胜平": 1,
                        "胜负": 2,
                        "平胜": 3,
                        "平平": 4,
                        "平负": 5,
                        "负胜": 6,
                        "负平": 7,
                        "负负": 8
                       }
point_res_dict = {
                  "0:0": 0,
                  "0:1": 1,
                  "0:2": 2,
                  "0:3": 3,
                  "0:4": 4,
                  "0:5": 5,
                  "1:0": 6,
                  "1:1": 7,
                  "1:2": 8,
                  "1:3": 9,
                  "1:4": 10,
                  "1:5": 11,
                  "2:0": 12,
                  "2:1": 13,
                  "2:2": 14,
                  "2:3": 15,
                  "2:4": 16,
                  "2:5": 17,
                  "3:0": 18,
                  "3:1": 19,
                  "3:2": 20,
                  "3:3": 21,
                  "4:0": 22,
                  "4:1": 23,
                  "4:2": 24,
                  "5:0": 25,
                  "5:1": 26,
                  "5:2": 27,
                  "胜其他": 28,
                  "平其他": 29,
                  "负其他": 30
                 }
MAX_GOALS = 7

class WebModel(object):
  def __init__(self, name):
    self.name = name
    self.parser = etree.HTMLParser()
    self.res_list = []
    self.home_url = "http://www.lottery.gov.cn"
    self.url = self.home_url + "/football/result.jspx"
    self.shouzhu_url = self.home_url + "/football/match_list.jspx"
    self.data_select_match_path = '//html/body/div[@class="main zsgkj"]/div[@class="xxsj"]/table/tr/td/table/tr/td/table/tr/td/div/div/select/option/@value'
    self.data_other_match_path = self.home_url + "/football/result_%s.jspx"
    self.match_link_path = '//html/body/div[@class="main zsgkj"]/div[@class="xxsj"]/table/tr/td/table/tr'
    self.selm_match_xpath = './td[5]/text()'
    self.full_match_xpath = './td[6]/text()'
    self.current_match_link_xpath = './td[8]/a/@href'
    # for each data
    self.data_main_path = '//html/body/div[@class="main zgdjj"]/'
    self.data_data_path = self.data_main_path + 'div[@class="xxsj"]/'
    # for info of each match
    self.data_info_path = self.data_main_path + 'div[@class="ggt yylClear"]/'
    self.data_home_team_name = self.data_info_path + 'div[@class="d1"]/text()'
    self.data_guest_team_name = self.data_info_path + 'div[@class="d3"]/text()'
    self.data_liga_name = self.data_info_path + 'div[@class="d2"]/p[@class="p1"]/text()'
    self.data_match_time = self.data_info_path + 'div[@class="d2"]/p[@class="p2"]/text()'
    # for result of each match
    self.data_res_path = self.data_data_path + '/table[1]/tr[3]/'
    self.data_match_result = self.data_res_path + 'td[2]/text()'
    self.data_match_rang_point = self.data_res_path + 'td[3]/text()'
    self.data_match_rang_result = self.data_res_path + 'td[4]/text()'
    self.data_match_point = self.data_res_path + 'td[5]/text()'
    self.data_match_goals = self.data_res_path + 'td[6]/text()'
    self.data_match_merge_result = self.data_res_path + 'td[7]/text()'
    # for win/draw/lose probability for each match
    self.data_prob_path = self.data_data_path + '/table[2]/tbody/tr'
    self.data_each_prob_turn = './td[1]/text()'
    self.data_each_prob_win = './td[2]/text()'
    self.data_each_prob_draw = './td[3]/text()'
    self.data_each_prob_lose = './td[4]/text()'
    # for rang qiu win/draw/lose probability for each match
    self.data_prob_rang_path = self.data_data_path + '/table[3]/tbody/tr'
    self.data_each_rang_prob_turn = './td[1]/text()'
    self.data_each_rang_prob_rang = './td[2]/text()'
    self.data_each_rang_prob_win = './td[3]/text()'
    self.data_each_rang_prob_draw = './td[4]/text()'
    self.data_each_rang_prob_lose = './td[5]/text()'
    # for goals probability for each match
    self.data_prob_goals_path = self.data_data_path + '/table[5]/tbody/tr'
    self.data_goal_turns = './td'
    # for merge probability for each match
    self.data_prob_merge_name_path = self.data_data_path + '/table[6]/tr/td'
    self.data_prob_merge_data_path = self.data_data_path + '/table[6]/tbody/tr'
    self.data_merge_turns = './td'
    # for points probability for each match
    self.data_prob_points_path = self.data_data_path + '/table[4]/tr'
    self.data_prob_points_title = './td/span/b/text()'
    self.data_prob_points_td_path = './td'
    self.data_prob_points_turn = './strong/span/text()'
    # shouzhu path
    self.shouzhu_select_match_path = '//html/body/div[@class="main zszsc"]/div[@class="xxsj"]/table/tr/td/table/tr/td/a/@href'

  def try_get_match(self, doc, path, has_result=False):
    res = doc.xpath(path)
    if len(res) <= 0:
      res = ""
      has_result = False or has_result
    else:
      res = res[0]
      has_result = True
    return res, has_result

  def process_info(self, doc, res_dict):
    res_dict_info = {}
    home_team_name, _ = self.try_get_match(doc, self.data_home_team_name)
    if not _:
      res_dict["info"] = res_dict_info
      return res_dict
    guest_team_name = doc.xpath(self.data_guest_team_name)[0]
    liga_name = doc.xpath(self.data_liga_name)[0]
    match_time = doc.xpath(self.data_match_time)[0]
    res_dict_info["home"] = home_team_name.split(" ")[0]
    res_dict_info["guest"] = guest_team_name.split(" ")[0]
    res_dict_info["liga"] = liga_name
    res_dict_info["match_time"] = match_time
    res_dict["info"] = res_dict_info
    return res_dict

  def judge_result_from_full(self, res, selm, full):
    if res:
      return res
    else:
      left, right = [int(x) for x in full.split(":")]
      if left > right:
        return u'胜'
      elif left == right:
        return u'平'
      else:
        return u'负'

  def judge_rang_from_full(self, res, rang, selm, full):
    if res:
      return res
    else:
      left, right = [int(x) for x in full.split(":")]
      if left + rang > right:
        return u'胜'
      elif left + rang == right:
        return u'平'
      else:
        return u'负'

  def judge_point_from_full(self, res, selm, full, rang):
    rang = int(rang[1:3])
    if res:
      return res
    else:
      if full in point_res_dict:
        return full
      else:
        left, right = [int(x) for x in full.split(":")]
        if left > right:
          return u'胜其他'
        elif left + rang == right:
          return u'平其他'
        else:
          return u'负其他'
        
  def judge_goals_from_full(self, res, selm, full):
    if res:
      return res
    else:
      left, right = [int(x) for x in full.split(":")]
      if (left + right) < MAX_GOALS:
        return str(left + right)
      else:
        return u'7+'
        
  def judge_merge_from_full(self, res, selm, full):
    if res:
      return res
    else:
      return self.judge_result_from_full("", "", selm) + self.judge_result_from_full("", "", full)
        
  def process_result(self, doc, res_dict, selm="", full=""):
    # res path
    res_dict_res = {}
    hr = False
    match_result, hr = self.try_get_match(doc, self.data_match_result, hr)
    match_rang_point, _ = self.try_get_match(doc, self.data_match_rang_point)
    match_rang_result, hr = self.try_get_match(doc, self.data_match_rang_result, hr)
    match_point, hr = self.try_get_match(doc, self.data_match_point, hr)
    match_goals, hr = self.try_get_match(doc, self.data_match_goals, hr)
    match_merge_result, hr = self.try_get_match(doc, self.data_match_merge_result, hr)
    if hr or full or selm:
      res_dict_res["match_result"] = self.judge_result_from_full(match_result, selm, full)
      res_dict_res["match_rang_point"] = match_rang_point
      res_dict_res["match_rang_result"] = self.judge_rang_from_full(match_rang_result, int(match_rang_point.replace("(", "").replace(")", "")), selm, full)
      res_dict_res["match_point"] = self.judge_point_from_full(match_point, selm, full, match_rang_point)
      res_dict_res["match_goals"] = self.judge_goals_from_full(match_goals, selm, full)
      res_dict_res["match_merge_result"] = self.judge_merge_from_full(match_merge_result, selm, full)
    res_dict["result"] = res_dict_res
    return res_dict

  def process_probability(self, doc, res_dict):
    # match result probability
    res_dict_prob = []
    prob = doc.xpath(self.data_prob_path)
    if len(prob) <= 0:
      res_dict["probability"] = res_dict_prob
      return res_dict
    for _, pr in enumerate(prob):
      res_dict_prob_turn = {}
      each_prob_turn = pr.xpath(self.data_each_prob_turn)[0]
      each_prob_win = pr.xpath(self.data_each_prob_win)[0]
      each_prob_draw = pr.xpath(self.data_each_prob_draw)[0]
      each_prob_lose = pr.xpath(self.data_each_prob_lose)[0]
      res_dict_prob_turn["turn"] = each_prob_turn
      res_dict_prob_turn["win"] = each_prob_win
      res_dict_prob_turn["draw"] = each_prob_draw
      res_dict_prob_turn["lose"] = each_prob_lose
      res_dict_prob.append(res_dict_prob_turn)
    res_dict["probability"] = res_dict_prob
    return res_dict

  def process_probability_rang_qiu(self, doc, res_dict):
    # match result rang qiu probability
    res_dict_prob_rang = []
    prob_rang = doc.xpath(self.data_prob_rang_path)
    for _, pr in enumerate(prob_rang):
      res_dict_prob_rang_turn = {}
      each_prob_turn = pr.xpath(self.data_each_rang_prob_turn)[0]
      each_prob_rang = pr.xpath(self.data_each_rang_prob_rang)[0]
      each_prob_win = pr.xpath(self.data_each_rang_prob_win)[0]
      each_prob_draw = pr.xpath(self.data_each_rang_prob_draw)[0]
      each_prob_lose = pr.xpath(self.data_each_rang_prob_lose)[0]
      res_dict_prob_rang_turn["turn"] = each_prob_turn
      res_dict_prob_rang_turn["rang_qiu_shu"] = each_prob_rang
      res_dict_prob_rang_turn["win"] = each_prob_win
      res_dict_prob_rang_turn["draw"] = each_prob_draw
      res_dict_prob_rang_turn["lose"] = each_prob_lose
      res_dict_prob_rang.append(res_dict_prob_rang_turn)
    res_dict["probability_rang_qiu"] = res_dict_prob_rang 
    return res_dict

  def process_probability_goals(self, doc, res_dict):
    # match result goals probability
    res_dict_prob_goals = []
    prob_goals = doc.xpath(self.data_prob_goals_path)
    for i, pr in enumerate(prob_goals):
      goal_turns = pr.xpath(self.data_goal_turns)
      res_dict_prob_goals_turn = {}
      for _, pr in enumerate(goal_turns):
        if _ == 0:
          cur_turn = pr.xpath('./text()')[0]
          res_dict_prob_goals_turn["turn"] = cur_turn
        elif _ < len(goal_turns) - 1:
          each_prob_goals = pr.xpath('./text()')[0]
          res_dict_prob_goals_turn[str(_ - 1)] = each_prob_goals
        else:
          each_prob_goals = pr.xpath('./text()')[0]
          res_dict_prob_goals_turn[str(_ - 1) + "+"] = each_prob_goals
      res_dict_prob_goals.append(res_dict_prob_goals_turn) 
    res_dict["probability_goals"] = res_dict_prob_goals
    return res_dict

  def process_probability_merge(self, doc, res_dict):
    # match merge result probability
    res_dict_prob_merge = []
    prob_merge = doc.xpath(self.data_prob_merge_name_path)
    # get merge name
    name_list = []
    for i, prm in enumerate(prob_merge):
      if 0 < i:
        name_list.append(prm.xpath('./text()')[0])
    prob_merge = doc.xpath(self.data_prob_merge_data_path)
    for i, prm in enumerate(prob_merge):
      res_dict_prob_merge_turn = {}
      merge_turns = prm.xpath(self.data_merge_turns)
      for _, prt in enumerate(merge_turns):
        if _ == 0:
          cur_turn = prt.xpath('./text()')[0]
          res_dict_prob_merge_turn["turn"] = cur_turn
        else:
          each_prob_merge = prt.xpath('./text()')[0]
          res_dict_prob_merge_turn[name_list[_]] = each_prob_merge
      res_dict_prob_merge.append(res_dict_prob_merge_turn)
    res_dict["probability_merge"] = res_dict_prob_merge
    return res_dict

  def process_probability_points(self, doc, res_dict):
    # match points probability
    res_dict_prob_points = []
    prob_points = doc.xpath(self.data_prob_points_path)
    res_dict_prob_points_turn = {}
    name_list = []
    beg_title = True
    for i, pr in enumerate(prob_points):
      if i == 0:
        prob_points_title = pr.xpath(self.data_prob_points_title)
      else:
        td_path = pr.xpath(self.data_prob_points_td_path)
        if len(td_path) <= 1:
          if res_dict_prob_points_turn:
            res_dict_prob_points.append(res_dict_prob_points_turn)
          res_dict_prob_points_turn = {}
          cur_turn = td_path[0].xpath(self.data_prob_points_turn)[0]
          beg_title = True
          res_dict_prob_points_turn["turn"] = cur_turn
          name_list = []
        else:
          for _, tdp in enumerate(td_path):
            if beg_title:
              cur_item_point = tdp.xpath('./text()')[0]
              if len(cur_item_point) <= 1:
                continue
              name_list.append(cur_item_point)
            else:
              cur_item_prob = tdp.xpath('./text()')[0]
              res_dict_prob_points_turn[name_list[_]] = cur_item_prob
          if not beg_title:
            name_list = []
          beg_title = not beg_title
    if res_dict_prob_points_turn:
      res_dict_prob_points.append(res_dict_prob_points_turn)
    res_dict["probability_points"] = res_dict_prob_points
    return res_dict
 
  def write_each_match(self, url, selm="", full=""):
    res_dict = {}
    print "cur_url:", url
    doc = etree.parse(url, self.parser)
    # process info
    res_dict = self.process_info(doc, res_dict)
    if not res_dict["info"]:
      return res_dict
    # process res
    res_dict = self.process_result(doc, res_dict, selm, full)
    # process probability
    res_dict = self.process_probability(doc, res_dict)
    # process probability rang qiu
    res_dict = self.process_probability_rang_qiu(doc, res_dict)
    # process probability goals
    res_dict = self.process_probability_goals(doc, res_dict)
    # process probability merge
    res_dict = self.process_probability_merge(doc, res_dict)
    # process probability points
    res_dict = self.process_probability_points(doc, res_dict)
    print "res_dict:", json.dumps(res_dict, ensure_ascii=False, indent=4, sort_keys=True).encode("utf-8")
    return res_dict

  def write_match(self, url):
    res_list = []
    doc = etree.parse(url, self.parser)
    match_path = doc.xpath(self.match_link_path)
    for i, mp in enumerate(match_path):
      if 0 < i:
        selm, _ = self.try_get_match(mp, self.selm_match_xpath)
        full, _ = self.try_get_match(mp, self.full_match_xpath)
        match_link, _ = self.try_get_match(mp, self.current_match_link_xpath)
        print "selm:", selm, "full:", full
        if match_link:
          match_link = self.home_url + match_link
          match_dict = self.write_each_match(match_link, selm, full)
          res_list.append(match_dict)
    return res_list

  def get_data(self):
    doc = etree.parse(self.url, self.parser)
    mid = doc.xpath(self.data_select_match_path)
    for m in mid:
      if int(m) == 1:
        self.res_list.extend(self.write_match(self.url))
      else:
        self.res_list.extend(self.write_match(self.data_other_match_path % m))
    return self.res_list

  def get_shouzhu_match(self):
    self.shouzhu_match_list = []
    doc = etree.parse(self.shouzhu_url, self.parser)
    mid = doc.xpath(self.shouzhu_select_match_path)
    print "len mid:", len(mid)
    for m in mid:
      print "cur html path:", self.home_url + m
      self.shouzhu_match_list.append(self.write_each_match(self.home_url + m))
    return self.shouzhu_match_list
