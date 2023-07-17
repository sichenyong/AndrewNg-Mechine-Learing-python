from scipy.io import loadmat
import numpy as np
import displayData
import costFunction
import oneVSall
import predictOneVsAll
import matplotlib.pyplot as plt
# =================== load data ===================
data = loadmat('./ex3/ex3data1.mat')
x_data = data['X'] # 5000 * 400
y_data = data.get('y') # 5000 * 1
# x_data = np.array([im.reshape((20, 20)).T for im in x_data]) # 因为矩阵里本来每行存储的是x的转置
# x_data = np.array([im.reshape((400, )) for im in x_data])
# print(y_data)
# =================== visualize the data ===================
rand = np.random.randint(0, 5000, (100, ))  # [0, 5000)
# displayData.data_display(x_data[rand,:])

# =================== Test case for costFunction ===================
theta_t = np.array([-2, -1, 1, 2])
t = np.linspace(1, 15, 15) / 10
t = t.reshape((3, 5))
x_t = np.column_stack((np.ones((5, 1)), t.T))
y_t = np.array([1, 0, 1, 0, 1])
l_t = 3
cost = costFunction.cost(theta_t, x_t, y_t, l_t)
grad = costFunction.gradient(theta_t, x_t, y_t, l_t)
print("cost is {}".format(cost))
print("expected cost is 2.534819")
print("grad is {}".format(grad))
print("expected grad is 0.146561 -0.548558 0.724722 1.398003")
# ============================ test end =============================================
lamda = 0.1
num_labels = 10
theta = oneVSall.one_VS_all(x_data,y_data,lamda,num_labels)
# print(theta.shape)
result = predictOneVsAll.pred(theta, x_data[1500, :])
np.set_printoptions(precision=2, suppress=True)  # don't use  scientific notation
print("this number is {}".format(result))  # 3
plt.imshow(x_data[1500, :].reshape((20, 20)), cmap='gray', vmin=-1, vmax=1)
plt.show()
accuracy = predictOneVsAll.pred_accuracy(theta, x_data, y_data)
print("test 5000 images, accuracy is {:%}".format(accuracy))