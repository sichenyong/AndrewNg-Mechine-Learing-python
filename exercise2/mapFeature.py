import numpy as np
def mapfeature(x_data):
    m = x_data.shape[0]
    temp_x = np.zeros((m, 27))
    x1 = x_data[:,0].reshape((-1,1))
    x2 = x_data[:,1].reshape((-1,1))
    end = 0
    for i in range(1, 7):
        for j in range(0, i + 1):
            temp_x[:, end] = np.multiply(np.power(x1[:, 0], i-j), np.power(x2[:, 0], j))
            end += 1
    return temp_x