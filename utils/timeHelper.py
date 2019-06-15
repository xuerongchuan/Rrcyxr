# -*- coding: utf-8 -*-

import datetime
import time

def getTime(timeVal):
  t = datetime.datetime.strptime(timeVal, '%Y-%m-%d')
  d = t.timetuple()
  return int(time.mktime(d))

def getDate(timestamp):
  timeArray = time.localtime(timestamp)
  timestr = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
  date = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
  return date