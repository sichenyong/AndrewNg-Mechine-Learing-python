import numpy as np
import plotData
from scipy.io import loadmat
import estimateGuassian
import multivariateGaussian
import selectThreshold

#================== Part 1: Load Example Dataset======================
data1 = loadmat("./machine-learning-ex8/ex8/ex8data1.mat")
Xval = data1["Xval"]
yval = data1["yval"]
X = data1["X"]

plotData.plto_2D(X,1,'The first dataset')
#================== Part 2: Estimate the dataset statistic ======================
mu, sigma2 = estimateGuassian.estimate(X)
print("μ is {}".format(mu))
print("σ is {}".format(sigma2))
p = multivariateGaussian.do(X,mu,sigma2)
plotData.visualizeFit(X, mu, sigma2,2,'The Gaussian distribution contours of the distribution fit to the dataset')
#================== Part 3: Find Outliers ======================
pval = multivariateGaussian.do(Xval,mu,sigma2)
epsilon, f1score = selectThreshold.selectThreshold(yval,pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('you should see a value epsilon of about 8.99e-05')
print('Best F1 on Cross Validation Set:  {}'.format(f1score))
print('you should see a Best F1 value of  0.875000')
outliers = np.where(p < epsilon)[0]
plotData.visualizeFit(X, mu,sigma2,3,'The classified anomalies',outliers=outliers)
#================== Part 4: Multidimensional Outliers ======================
data2 = loadmat("./machine-learning-ex8/ex8/ex8data2.mat")
X = data2["X"]
Xval = data2["Xval"]
yval = data2["yval"]
# get  Gaussian parameters 
mu, sigma2 = estimateGuassian.estimate(X)
# evaluate the probabilities for the training set X
p = multivariateGaussian.do(X,mu,sigma2)
# evaluate the probabilities for the cross-validation set Xval
pval = multivariateGaussian.do(Xval,mu, sigma2)
epsilon, f1score = selectThreshold.selectThreshold(yval, pval)
print("\n\n ========== high dimension dataset ========== ")
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('you should see a value epsilon of about 1.38e-18')
print('Best F1 on Cross Validation Set:  {}'.format(f1score))
print('you should see a Best F1 value of   0.615385')
print('outliers found:{}'.format(sum(p < epsilon)))
print('you should see a correct value of  117') 