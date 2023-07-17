from scipy.io import loadmat
import numpy as np
import predict
import displayData
# =================load data=================
data = loadmat('./ex3/ex3data1.mat')
x_data = data['X'] # 5000 * 400
y_data = data.get('y') # 5000 * 1
# x_data = np.array([im.reshape((20, 20)).T for im in x_data]) # 必须转置才能得到正确图像
# x_data = np.array([im.reshape((400, )) for im in x_data])
# =================load weight=================
weights = loadmat('./ex3/ex3weights.mat')
theta1 = weights['Theta1'] #25 401
theta2 = weights['Theta2'] # 10 26
# =================predict=================
hx = predict.predict(theta1,  theta2, x_data)
right_num = 0

for i in range(hx.shape[0]):
    if hx[i] == y_data[i,:]:
        right_num += 1

accuracy = right_num / hx.shape[0]
print("test 5000 images accuracy is {:%}".format(accuracy))
print("expected result is 97.5%")

