

import time
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


class KDD(object):

	def __init__(self, config, dl):

		self.config = config
		self.dl = dl
		self.mat = self.dl.train_sp_matrix.tocsr()
		self.build_model()

	def _compute_distance(self, x, y):

		if self.config.distance == 'cosine':
			return self._cosine_similarity(x,y)
	def _cosine_similarity(self,X,Y):

		'''
		输入：x - 1xn 维离散稀疏向量
		y - nx1 维离散稀疏向量
		'''
		if self.config.implicit:
			return X*Y/np.sqrt((X.sum(1)*Y.sum(0)))
		else:
			return x*y/np.sqrt((x.power(2).mean()*y.power(2).mean()))

	def generate_similarity_matrix(self):
		
		if self.config.item:
			print('正在计算物品相似度矩阵')
			st = time.time()
			mat = self.mat.transpose()

			self.similar_mat = sp.csr_matrix(self._compute_distance(mat, mat.transpose()))
			et = time.time()
			print('物品相似度矩阵计算完毕，用时%.4fs'%(et-st))
		else:
			print('正在计算用户相似度矩阵')
			st = time.time()
			self.similar_mat = sp.csr_matrix(self._compute_distance(self.mat, self.mat.transpose()))
			et = time.time()
			print('用户相似度矩阵计算完毕，用时%.4fs'%(et-st))
	def build_model(self):
		self.generate_similarity_matrix()


	def predict(self, u, j):

		u_hist = self.mat.getrow(u).indices
		
		similars = sorted(zip(self.similar_mat.getrow(j).indices, self.similar_mat.getrow(j).data), \
			key=lambda x:x[1] , reverse=True)[:self.config.K]
		p_u_j = sum([v for i,v in similars if i in u_hist])
		
		return p_u_j
	def recommend_for_u(self, u):
		rank = defaultdict(float)
		u_hist = self.mat.getrow(u).indices
		for i in u_hist:
			similars = sorted(zip(self.similar_mat.getrow(i).indices, self.similar_mat.getrow(i).data), \
			key=lambda x:x[1] , reverse=True)[:self.config.K]
			for j, w in similars:
				if j in u_hist:
					continue
				rank[j] += w
		recommends = sorted(rank.items(), key=lambda x:x[1], reverse=True)[:self.config.N]
		return recommends



	def evaluate(self):
		st = time.time()
		hit = 0
		pre_len = 0
		recall_len = 0
		users , items = zip(*self.dl.test_list)
		for u, items  in self.dl.test_list:
			recommends = self.recommend_for_u(u)
			for i,v in recommends:
				if i in items:
					hit +=1
			pre_len +=self.config.N
			recall_len+=len(items)
		et = time.time()

		precision, recall = hit/pre_len, hit/recall_len
		print('hit: %d, precision:%.4f, recall:%.4f, time:%.4fs'%(hit, precision, recall, (et-st)))



