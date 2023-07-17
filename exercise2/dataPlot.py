import matplotlib.pyplot as plt
import numpy as np
import mapFeature
# flag 用于标识是否画出decision boundary
def Plot(x_data, y_data,theta = None,flag = False, legend = ["Admitted", "Not admitted"]):
    pos = np.where(y_data == 1)
    neg = np.where(y_data == 0)

    plt.plot(x_data[pos[0],0], x_data[pos[0],1],'k+')
    plt.plot(x_data[neg[0],0], x_data[neg[0],1],'ko', color="y")
    plt.xlabel('exam1 score')
    plt.ylabel('exam2 score')
    plt.legend(legend, loc='upper right')
    if flag:
        if x_data.shape[1] == 3:
            plot_x = np.zeros((2,))
            plot_x[0] = np.min(x_data[:,1])
            plot_x[1] = np.max(x_data[:,2])
            plot_y = -(theta[1]*plot_x+theta[0])/theta[2]
            plt.plot(plot_x, plot_y)
            plt.show()
        elif x_data.shape[1] > 3:
            u = np.linspace(-1, 1.5, 50)
            v = np.linspace(-1, 1.5, 50)
            z = np.ones((u.shape[0], v.shape[0]))
            for i in range(u.shape[0]):
                for j in range(v.shape[0]):
                    tempt = np.column_stack((np.ones((1, 1)), mapFeature.mapfeature(x1 = np.array([u[i]]), x2 = np.array([v[j]]))))
                    a = np.dot(tempt, theta)
                    z[i, j] = a
            plt.contour(u, v, z, [0], colors='k')
            plt.show()
    else:
        plt.show()
    return plt

