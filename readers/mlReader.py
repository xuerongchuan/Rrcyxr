import json
import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
from dataIO.splits import random_split

class Dataloader(object):
	'''
	导入ml-1m的数据
	数据要求：
	1.用户商品id都经过映射从0开始
	'''
	def __init__(self, config):
		self.config = config
		self.num_items = 3883 #数据预处理时计算的
		self.num_users = 6040
		self._init_data()

	def _init_data(self):
		if self.config.month_data:
			train = pd.read_csv(self.config.train_path)
			test = pd.read_csv(self.config.test_path)
		else:
			data = pd.read_csv(self.config.data_path)
			train, test = random_split(data, 0.2)
		if self.config.implicit:
			self.train_sp_matrix = sp.coo_matrix(([1]*len(train), [train.user.values, \
			train.movie.values]))
		else:
			self.train_sp_matrix = sp.coo_matrix((train.rating.values, [train.user.values, \
			train.movie.values]))
		self.test_list = self._generate_data_list(test)
	def _generate_data_list(self, data):
		data_list = []
		for u in data.user.unique():
			u_items = list(data[data.user==u].movie.values)
			data_list.append((u, u_items))
		return data_list






