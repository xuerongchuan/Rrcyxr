import os
class Config(object):

	def __init__(self):
		self.test = 0
		self.month_data =1
		self.data_path = '../../data/ml-1m/'
		self.train_path = '../data/ml-1m/train-10'
		self.test_path = '../data/ml-1m/test-10'
		self.distance = 'cosine'
		self.implicit = 1
		self.item = 1
		self.k = 50 #k近邻
		self.n = 10 # top-n
		self.sep = ','
		print(os.getcwd())