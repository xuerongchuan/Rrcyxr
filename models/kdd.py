
class KDD(object):

	def __init__(self, config, dl):

		self.config = config
		self.dl = dl

	def _compute_distance(self, x, y):

		if self.config.distance == 'cosine':
			return self._cosine_similarity(x,y)
	def _cosine_similarity(self,x,y):

		'''
		输入：x - 1xn 维离散稀疏向量
		y - nx1 维离散稀疏向量
		'''
		if self.config.implicit:
			return x*y/(x.sum()*y.sum())
		else:
			return x*y/(x.power(2).mean()*y.power(2).mean())

	def generate_similarity_matrix(self):
		
		if self.config.item:
			self.mat = self.mat.transpose()

