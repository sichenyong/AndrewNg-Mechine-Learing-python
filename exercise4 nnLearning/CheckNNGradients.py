import numpy as np
import debugInitializeWeights
import nnCostFunction

def grad_numerically(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels):
    # compute grad numerically
    epsilon = 1e-4
    grad_num = np.zeros(nn_params.shape)
    for i in range(nn_params.size):
        nn_params_temptl = nn_params.copy()
        nn_params_temptr = nn_params.copy()
        nn_params_temptl[i] -= epsilon
        nn_params_temptr[i] += epsilon
        jl = nnCostFunction.nn_cost(nn_params_temptl, input_layer_size, hidden_layer_size, X, y, num_labels)
        jr = nnCostFunction.nn_cost(nn_params_temptr, input_layer_size, hidden_layer_size, X, y, num_labels)
        grad_num[i] = (jr - jl) / 2 / epsilon
    return grad_num

def check():
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights.initialize(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights.initialize(num_labels, hidden_layer_size)
    X = debugInitializeWeights.initialize(m, input_layer_size - 1)
    y = 1 + np.mod(np.array(range(1, m+1, 1)), num_labels)
    nn_params = np.hstack((Theta1.reshape((1,-1)),Theta2.reshape((1,-1)))).flatten()
    grad1 = nnCostFunction.backpropagation(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels)
    grad2 = grad_numerically(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels)
    print("grad自己的算法:{}".format(grad1))
    print("gradN:{}".format(grad2))

def grad_numerically_reg(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels,l):
    # compute grad numerically
    epsilon = 1e-4
    grad_num = np.zeros(nn_params.shape)
    for i in range(nn_params.size):
        nn_params_temptl = nn_params.copy()
        nn_params_temptr = nn_params.copy()
        nn_params_temptl[i] -= epsilon
        nn_params_temptr[i] += epsilon
        jl = nnCostFunction.nn_cost_reg(nn_params_temptl, input_layer_size, hidden_layer_size, X, y, num_labels,l)
        jr = nnCostFunction.nn_cost_reg(nn_params_temptr, input_layer_size, hidden_layer_size, X, y, num_labels,l)
        grad_num[i] = (jr - jl) / 2 / epsilon
    return grad_num

def check_reg(l = 1):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights.initialize(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights.initialize(num_labels, hidden_layer_size)
    X = debugInitializeWeights.initialize(m, input_layer_size - 1)
    y = 1 + np.mod(np.array(range(1, m+1, 1)), num_labels)
    nn_params = np.hstack((Theta1.reshape((1,-1)),Theta2.reshape((1,-1)))).flatten()
    grad1 = nnCostFunction.backpropagation_reg(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels,l)
    grad2 = grad_numerically_reg(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels,l)
    print("grad自己的算法:{}".format(grad1))
    print("gradN:{}".format(grad2))