import numpy as np

# 高斯核函数计算
def GuassianKernel(x, l,sigma=2):
    x = x.reshape((-1,1))
    l = l.reshape((-1,1))
    fz = x - l
    fz = np.sum(np.power(fz,2))
    fm = 2 * sigma * sigma
    return np.array([np.exp(-1 * fz / fm)])
