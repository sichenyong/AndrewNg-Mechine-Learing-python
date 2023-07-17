import numpy as np
from scipy.io import loadmat
import displayData
import nnCostFunction
import sigmoidGradient
import randInitializeWeights
import CheckNNGradients
import scipy.optimize as op
import Accuracy

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
# ======================= Loading data and visualizing =================================
data = loadmat("./ex4/ex4data1.mat")
X = data["X"]
y_data = data["y"]
x_data = [im.reshape((20,20)).T for im in X]
x_data = np.array([im.reshape((1,-1)) for im in x_data])
# randperm = np.random.randint(0, 5000, (100, ))
# display_data = x_data[randperm, :]
# displayData.display(display_data)
# ===================== load end ========================================================
nn_data = loadmat("./ex4/ex4weights.mat")
theta1 = nn_data["Theta1"] # 25 * 401
theta2 = nn_data["Theta2"] # 10 * 26
# ===================== test cost =======================================================
# unroll_params
nn_params = np.hstack((theta1[:].reshape((1, -1)), theta2[:].reshape((1, -1)))).flatten()
J = nnCostFunction.nn_cost(nn_params, input_layer_size, hidden_layer_size, X, y_data, num_labels)
print("cost is {}".format(J))
print("expected cost is 0.287629")

l = 1
J_Reg = nnCostFunction.nn_cost_reg(nn_params, input_layer_size, hidden_layer_size, X, y_data, num_labels, l)
print("cost with reg is {}".format(J_Reg))
print("expected cost is 0.383770")
# ===================== test end =======================================================
# ===================== sigmoid gradient =======================================================
g = sigmoidGradient.sigmoid_gradient(np.array([0,0,0]))
print("grad is {}".format(g))
print("excepted grad is 0.25, 0.25,0.25")
# ===================== sigmoid gradient end =======================================================
initial_theta1 = randInitializeWeights.initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights.initialize_weights(hidden_layer_size, num_labels)
# unroll params
initial_nn_params = np.hstack((initial_theta1[:].reshape((1,-1)),initial_theta2[:].reshape((1,-1)))).flatten()
# =================== implements BackPropagation =============================================
# check no reg
# CheckNNGradients.check()
# check with reg
# CheckNNGradients.check_reg()
result = op.minimize(nnCostFunction.nn_cost_reg,initial_nn_params,args=(input_layer_size,hidden_layer_size,x_data,y_data,num_labels,l),method="TNC",
                     jac=nnCostFunction.backpropagation_reg,options={"maxiter" : 400})
final_theta = result.x
final_theta1 = final_theta[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
final_theta2 = final_theta[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, -1))
print(result)
accuracy = Accuracy.cal_accuracy(x_data, y_data, final_theta1, final_theta2)
print("accuracy is {}%".format(accuracy * 100))
print("excepted accuracy is bettwen 94.3% and 96.3%")
