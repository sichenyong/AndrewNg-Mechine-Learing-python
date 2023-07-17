import numpy as np
import costFunction
def pred(theta, x_data):
    x_data = x_data.reshape((1,-1))
    x_data = np.column_stack((np.ones((1, 1)), x_data))
    result = costFunction.sigmoid(np.dot(x_data, theta.T)) # result 1 * m 看哪个概率最大
    predict = np.argmax(result.flatten())
    if predict == 0:
        predict = 10
    return predict

def pred_accuracy(theta, x_data, y_data):
    right_num = 0
    m, _ = x_data.shape
    for i in range(m):
        pd = pred(theta, x_data[i, :])
        if pd == y_data[i, :]:
            right_num += 1
    return right_num / m
