
class Config(object):

	def __init__(self):
		self.month_data =1
		self.data_path = '../../data/ml-1m/rating.csv'
		self.train_path = '../../data/ml-1m/train_month.csv'
		self.test_path = '../../data/ml-1m/test_month.csv'
		self.distance = 'cosine'
		self.implicit = 1
		self.item = 1
		self.K = 50
		self.N = 10