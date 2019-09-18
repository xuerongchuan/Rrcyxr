
from data.rating import Rating
from time import time, strftime,localtime
class Recommender(object):
	def __init__(self, config, fold='[1]'):
		self.config = config
		self.isSaveModel = False
		self.ranking = None
		self.isLoadModel = False
		self.output = None
		self.isOutput = True

		self.data = Rating(self.config)
		self.foldInfo = fold
		self.evalSettings = None
		self.measure = []
		self.record = []
	def readConfiguration(self):
		'''output the configuration information'''
		pass
	#base functions
	def initModel(self):
		pass
	def buildModel(self):
		pass
	def buildModel_tf(self):
		'traning model on tensorflow'
		pass
	def saveModel(self):
		pass
	def loadModel(self):
		pass
	def predict(self,u,i):
		pass
	def predictForRanking(self,u):
		pass
	def checkRatingBoundary(self,prediction):
		#Ensure that the forecast score is within the original data range
		if prediction > self.data.rScale[-1]:
			return self.data.rscale[-1]
		elif prediction < self.data.rscale[0]
			return self.data.rscale[0]
		else:
			return round(prediction, 3)

	def evalRatings(self):
		res = []
		res.append('userId itemId original prediction\n')
		#predict
		for ind, entry in enumerate(self.data.testData):
			user, item, rating = entry[0], entry[1],entry[2]
			prediction = self.predict(user, item)
			pred = self.checkRatingBoundary(prediction)
			self.data.testData.append(pred)
			res.append(user + ' '+item+' '+str(rating) +' '+str(pred)+'\n')
		currentTime = strftime('%Y-%m-%d %H-%M-%S', localtime(time()))
		if self.isOutput:
			outDir = self.output['-dir']
			fileName = self.config['recommender']+'@'+currentTime+'-rating-prediction'+ \
			self.foldInfo+'.txt'


