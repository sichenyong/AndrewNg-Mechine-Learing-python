import matplotlib.pyplot as plt
import numpy as np
import multivariateGaussian

def plto_2D(x_data,figure_num,title):
    plt.figure(num=figure_num)
    plt.title(title)
    latency = x_data[:,0]
    Throughput = x_data[:,1]
    plt.scatter(latency,Throughput,color = "blue", marker="x")
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

def visualizeFit(x_data, mu, sigma2,figure_num,title,outliers = None):
    plt.figure(num=figure_num)
    plt.title(title)
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian.do(np.column_stack((X1.flatten(), X2.flatten())), mu, sigma2)
    Z = Z.reshape(X1.shape)
    plt.scatter(x_data[:,0], x_data[:,1],color = "blue", marker='x')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    if np.isinf(Z).sum() == 0:
        levels = np.power(10.0, np.arange(-20, 1, 3))
        plt.contour(X1, X2, Z, levels)
    if outliers is not None:
        plt.scatter(x_data[outliers,0], x_data[outliers,1],facecolors = "none",edgecolors='red', linewidths=2,s=100)
    plt.show()
