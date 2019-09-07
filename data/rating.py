import numpy as np

class Rating(object):
	'''data access control'''
	def __init__(self, config):
		self.config = config
		self.ratingConfig = 