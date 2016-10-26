# Inspired from Karpathy's 2d toy example on neural nets

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_data(N=100,K=3,s=0.3):
	"""
	Input:
		N: Number of data points per class
		K: Number of classes
		s: A parameter for randomness. Keep it less than 0.5.
	Output:
		X,y: 2-Dimensional 3-class Data along with labels
	"""
	np.random.seed(0)
	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes

	X = np.zeros((N*K,D))
	y = np.zeros(N*K, dtype='uint8')

	for j in xrange(K):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N) # radius
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*s # theta
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ix] = j
	#fig = plt.figure()
	#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	#plt.xlim([-1,1])
	#plt.ylim([-1,1])
	#plt.show()
	return X,y