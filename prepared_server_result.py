#!/usr/bin/dev python
#coding=utf-8
#authou: spgoal
#date: Feb 23th, 2018

import os
import sys
import time
import xlwt
from reuse_cnn import *
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def get_res_job(date="", drop_prob=0.0, struct="dqn"):
  if not date:
    current_result = reuse_cnn_for_all(drop_prob=drop_prob, struct=struct)
  else:
    current_result = reuse_cnn_for_all(date=date, drop_prob=drop_prob, struct=struct)
  return current_result

all_list = ["", ""]
str_list = ["dqn", "cnn"]
res_list = []
for ol, st in zip(all_list, str_list):
  if not ol:
    fn = "cur_result_for_server_%s.txt" % st
  else:
    fn = "cur_result_for_server_%s_%s.txt" % (ol, st)
  with open(fn, "w") as rf:
    current_result = get_res_job(ol, struct=st)
    res_list.append(current_result)
    rf.write(json.dumps(current_result, ensure_ascii=False, indent=4, sort_keys=True))

def write_list_to_xls():
  wb = xlwt.Workbook(encoding="utf-8")
  st = wb.add_sheet("predict_result")
  cur_row, cur_col = 0, 0
  # first line
  st.write(cur_row, cur_col, "Match Info")
  cur_col += 1
  for ol, stt in zip(all_list, str_list):
    if not ol:
      con = "Default_%s" % stt
    else:
      con = ol + "_%s" % stt
    st.write_merge(cur_row, cur_row, cur_col, cur_col + 3, con)
    cur_col += 4
  cur_row += 1
  # write real content
  if not res_list:
    print "Sorry, there is no result list prepared."
    return
  else:
    match_len = len(res_list[0])
    first_dict = res_list[0]
    ks = sorted(res_list[0].keys())
    for r in range(match_len):
      cur_col = 0
      ck = ks[r]
      print "ck:", ck
      st.write(cur_row, cur_col, ck)
      cur_col += 1
      for i, res in enumerate(res_list):
        if ck in res:
          st.write(cur_row, cur_col, res[ck][0]["translation"])
          st.write(cur_row, cur_col + 1, max(res[ck][0]["probability"]))
          st.write(cur_row, cur_col + 2, res[ck][1]["translation"])
          st.write(cur_row, cur_col + 3, max(res[ck][1]["probability"]))
        cur_col += 4
      cur_row += 1
  with open("cur_result_for_server.xls", "w") as of:
    wb.save(of)

# add table
if __name__ == "__main__":
  write_list_to_xls()
