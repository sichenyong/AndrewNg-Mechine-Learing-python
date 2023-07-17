from scipy.io import loadmat
import numpy as np
import plotData
import svmTrain
import visualizeBoundaryLinear
import GuassianKernel
from sklearn import svm
# ========= Part1:loading data and visualizing data ==========================
data = loadmat("./ex6/ex6data1.mat")
# for i in data.keys():
#     print(i)
x_data = data["X"]
y_data = data["y"]
# print(x_data)
# print(y_data)
plotData.plot_2D(x_data, y_data,1)

# ========= Part2:train linear SVM ===========================================
# use sklearn, different from the lecture
C = 1
linear_svm = svmTrain.train(C)
linear_svm.fit(x_data, y_data.flatten())
title = "use linear kernel to fit linear regression, c=1"
visualizeBoundaryLinear.visualize_boundary(x_data, y_data,linear_svm,0,4,1.5,5,title,2)

C = 100
linear_svm = svmTrain.train(C)
linear_svm.fit(x_data, y_data.flatten())
title = "use linear kernel to fit linear regression, c=100"
visualizeBoundaryLinear.visualize_boundary(x_data,y_data,linear_svm,0,4.5,1.5,5,title,3)
# ============== Part 3: Implementing Gaussian Kernel ===============
# test guassianKernel
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
test_result = GuassianKernel.GuassianKernel(x1,x2,sigma)
print("test result is {}".format(test_result))
print("expected value is 0.324652")
# =============== Part 4: Visualizing Dataset 2 ================
data2 = loadmat("./ex6/ex6data2.mat")
x_data2 = data2["X"]
y_data2 = data2["y"]
plotData.plot_2D(x_data2, y_data2,4)
# use custom kernel !!!! some errors happened
# guassian_svm = svm.SVC(C = 1, kernel=GuassianKernel.GuassianKernel)
# guassian_svm.fit(x_data2, y_data2.flatten())
# title = "use custom guassian kernel to fit non-linear"
# visualizeBoundaryLinear.visualize_boundary(x_data2, y_data2,guassian_svm,0,1,0.4,1)
sigma = 0.1
gamma = np.power(sigma, -2.)
gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
gaus_svm.fit(x_data2, y_data2.flatten())
visualizeBoundaryLinear.visualize_boundary(x_data2,y_data2,gaus_svm, 0, 1, .4, 1.0,message="test",figure_num=5)
# =============== Part 6: Visualizing Dataset 3 ================
data3 = loadmat("./ex6/ex6data3.mat")
x_data3 = data3["X"]
y_data3 = data3["y"]
xval = data3["Xval"]
yval = data3["yval"]
plotData.plot_2D(x_data3, y_data3,6)
values = [0.01,0.03,0.1,0.3,1,3,10,30]
best_svm = 0
best_score = 0
best_c = 0
best_sigma = 0
best_error = None
for c in values:
    for sigma in values:
        gamma = np.power(sigma,-2.0)
        g_svm = svm.SVC(C=c, kernel="rbf",gamma=gamma)
        g_svm.fit(x_data3, y_data3.flatten())
        score = g_svm.score(xval, yval.flatten())
        prediction = g_svm.predict(xval)
        prediction = prediction.reshape((-1,1))
        prediction = prediction - yval
        if score > best_score:
            best_score = score
            best_svm = g_svm
            best_c = c
            best_sigma = sigma
            best_error = np.mean(prediction)
        
print("best scores:{} with c={},sigma={},error={}".format(best_score, best_c, best_sigma,best_error))
title = "best scores:{} with c={},sigma={},error={}".format(best_score, best_c, best_sigma,best_error)
visualizeBoundaryLinear.visualize_boundary(x_data3,y_data3,best_svm,-0.6,0.3,-0.8,0.6,title,7)