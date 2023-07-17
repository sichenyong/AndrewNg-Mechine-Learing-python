import dataPlot
import numpy as np
import sigmoid
import costFunction
import scipy.optimize as op
import matplotlib.pyplot as plt
import predict
#========================read datas========================
with open('./ex2/ex2data1.txt','r') as file:
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
        
dataPlot.Plot(x_data, y_data)
#========================initialize========================
x_data = np.column_stack((np.ones((m, 1)), x_data))
initial_theta = np.zeros((feature_number+1, 1)).flatten()  # must be a vector

#========================compute cost and gradient========================
cost = costFunction.cost_function(initial_theta, x_data,y_data)
print("initial_theta cost is {} (approx)".format(cost))
print("expected cost is 0.693")

grad = costFunction.gradient(initial_theta, x_data, y_data)
print("initial_theta grad is {}".format(grad))
# print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
#========================Optimizing using fminunc(matlab)/scipy(python)========================
result = op.minimize(costFunction.cost_function, initial_theta.flatten(), args=(x_data, y_data), method="TNC", jac=costFunction.gradient)
final_theta = result.x
# print(final_theta)
dataPlot.Plot(x_data,y_data,final_theta ,flag= True)

hx, res = predict.predict(final_theta.flatten(),[[1,45,85]])
print("[45,85]'s result is {}".format(hx))
print("expected result is 0.776")
print("admission? {}".format(res))
