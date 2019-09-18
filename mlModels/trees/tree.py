import numpy as np

class Node:
	def __init__(self,left,right,rule):
		self.left = left
		self.right = right
		self.feature = rule[0]
		self.threshold = rule[1]

class Leaf:
	def __init__(self, value):
		self.value = value
class DecisionTree(object):
	def __init__(
		self,
		classifier=True, 
		max_depth=None,
		n_feats=None,
		criterion='entropy',
		seed=None
		):
		if seed:
			np.random.seed(seed)
		self.depth = 0
		self.root = None

		self.n_feats = n_feats
		self.criterion = criterion
		self.classifier = classifier
		self.max_depth = max_depth if max_depth else np.inf

		if not classifier and criterion in ['gini','entropy']:
			raise ValueError(
				"{} is valid criterion only when classifier is True".format(criterion)
				)
		if classifier and criterion == 'mse':
			raise ValueError(
				"{} is valid criterion only when classifier is False".format(criterion)
				)

	def fit(self, X, Y):
		'''
		Fit a binary secision tree to a dataset.

		Parameters:
		----------
		X:(N,M): the training data whih N examples and M features.
		Y:(N,): An array of integer class labels for each example in X if 
		self.classifier = True,otherwise the set of target values for
		each example in X

		'''
		self.n_classes = max(Y) + 1 if self.classifier else None
		self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
		self.root = self._grow(X,Y)

	def _grow(self,X,Y):
		' grow a decision tree'
		# if all labels are same, return a leaf
		if len(set(Y)) == 1:
			if self.classifier:
				prob = np.zeros(self.n_classes)
				prob[Y[0]] = 1.0
			return Leaf(prob) if self.classifier else Leaf(Y[0])
		# If we have reached max_depth, return a leaf
		if self.depth >= max_depth:
			v = np.mean(Y, axis=0)
			if self.classifier:
				v = np.bincount(Y, minlength=self.n_classes)
			return Leaf(v)

		N,M = X.shape
		self.depth +=1
		feat_idxs = np.random.choice(M, self.n_feats, replace=False)

		#greedily selcet the best split according to'criterion'
		feat, thresh = self._segment(X, Y,feat_idxs)
		l = np.argwhere(X[:, feat] <= thresh).flatten()
		r = np.argwhere(X[:, feat] > thresh).flatten()

		# grow the childeren that result from the split
		left = self._grow(X[l,:], Y[l])
		right = self._grow(X[r, :], Y[r])
		return Node(left,right,(feat, thresh))

	def _segment(self, X, Y, feat_idxs):
		'''
		Find the optimal split rule (feature index and split threshold) for the 
		data according to 'criterion' 
		'''
		best_gain = -np.inf
		split_idx, split_thresh = None, None
		for i in feat_idxs:
			vals = X[:,i]
			levels = np.unique(vals) #the number of the values available for each features
			thresholds = (levels[:-1], levels[1:])/2
			gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

			if gains.max() > best_gain:
				split_idx = i
				best_gain = gains.max()
				split_thresh = thresholds[gains.argmax()]
		return split_idx, split_thresh

	def _impurity_gain(self,Y, split_thresh, feat_vals):

		if self.criterion == 'entropy':
			loss =entropy
		if self.criterion == 'gini':
			loss = gini
		if self.criterion == 'mse':
			loss = mse
		base_loss = loss(Y)
		#split
		left = np.argwhere(feat_vals <=split_thresh).flatten() 
		right = np.argwhere(feat_vals > split_thresh).flatten()

		if len(left)==0 or len(right)==0:
			return 0

		n = len(Y)
		n_l, n_r = len(left), len(right)
		e_l, e_r = loss(Y[left]), loss(Y[right])
		child_loss = (n_l/n)*e_l + (n_r/n)*e_r

		ig = base_loss - child_loss
		return ig
	def _prune(self):
		child_tree = self.root
		if self.criterion == 'gini':
			loss = gini
		if self.criterion == 'mse':
			loss = mse
		if self.criterion == 'entropy':
			raise ValueError('criterion must be gini or mse')
	def _compute_alpha(self, node, loss):
		if node.isinstance(Leaf):
			return 
		alpha = min(loss)

	def predict(self, X):
		return np.array([self._traverse(x, self.root) for x in X])
	def predict_calss_probs(self,X):
		assert self.classifier
		return np.array([self._traverse(x, self.root, prob=True) for x in X])
	def _traverse(self, X, node, prob= False):
		if isinstance(node, Leaf):
			if self.classifier:
				return node.value if prob else node.value.argmax()
			return node.value
		if X[node.feature] <= node.threshold:
			return self._traverse(X, node.left, prob)
		return self._traverse(X, node.right, prob)

def entropy(y):
	hist = np.bincount(y)
	ps = hist/np.sum(hist)
	return -np.sum([p * np.log2(p) for p in ps if p >0])
def mse(y):
	return np.mean((y-np.mean(y))**2)
def gini(y):
	hist = np.bincount(y)
	N = np.sum(hist)
	return 1-sum([(i/N)**2 for i in hist])


