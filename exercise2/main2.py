import numpy as np
import dataPlot
import mapFeature
import costFunction_reg
import scipy.optimize as op
#========================read datas========================
with open('./ex2/ex2data2.txt','r') as file:
    lines = file.readlines()
    m = len(lines)
    feature_number = len(lines[0].strip().split(',')) - 1
    x_data = np.zeros((m,feature_number))
    y_data = np.zeros((m,1))
    for i in range(m):
        line_temp = lines[i].strip().split(',')
        for j in range(len(line_temp)):
            if j != len(line_temp) - 1:
                x_data[i,j] = line_temp[j]
            else:
                y_data[i,0] = line_temp[j]
        
dataPlot.Plot(x_data, y_data, legend=["y = 1", "y = 0"])

x_data = mapFeature.mapfeature(x_data)
x_data = np.column_stack((np.ones((m,1)),x_data))
initial_theta = np.ones(x_data.shape[1]).flatten()
lamda = 10
cost = costFunction_reg.cost_function(initial_theta,x_data,y_data,lamda)
print("cost is {}".format(cost))
print("expected cost is 3.16")
grad = costFunction_reg.gradient(initial_theta, x_data, y_data, lamda)
print("grad is {}".format(grad[0:5]))
print("0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n")

result = op.minimize(costFunction_reg.cost_function, initial_theta, args=(x_data, y_data, lamda), method="TNC", jac=costFunction_reg.gradient)
# print(result)
final_theta = result.x
# dataPlot.Plot(x_data[:,1:], y_data,final_theta,flag=True,legend=["y = 1", "y = 0"])