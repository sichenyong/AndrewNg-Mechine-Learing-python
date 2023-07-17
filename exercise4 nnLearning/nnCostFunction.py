import numpy as np
from scipy.special import expit
import sigmoidGradient
# hx and y are real numbers
def cost_function(hx, y):
    return -y * np.log(hx) - (1 - y) * np.log(1 - hx)

# don't use regularization
def nn_cost(nn_params, input_layer_size, hidden_layer_size,x_data, y_data ,num_labels):
    y_data = y_data.reshape((-1,1))
    theta1 = nn_params[0: hidden_layer_size * (input_layer_size+1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size+1):].reshape((num_labels, -1))

    # use loop for forward propagation
    sum_cost = 0
    m = x_data.shape[0]
    for i in range(m):
        a1 = x_data[i,:].reshape((-1,1))
        label = y_data[i,:] # 1~10
        a1 = np.vstack((np.ones((1,1)), a1)) # add bias unit
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1,1)), a2))
        z3 = theta2 @ a2
        a3 = expit(z3) # hx

        cost = 0
        # j : 0~9, j+1: 1 ~ 10
        for j in range(a3.shape[0]):
            if j+1 == label:
                flag = 1
            else:
                flag = 0
            cost += cost_function(a3[j,:], flag)
        sum_cost += cost
    sum_cost /= m
    return sum_cost

# use regularization
def nn_cost_reg(nn_params, input_layer_size, hidden_layer_size,x_data, y_data ,num_labels, l):
    sum_cost = nn_cost(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels)
    m = x_data.shape[0]
    theta1 = nn_params[0: hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, -1))
    regularization = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))
    regularization = regularization * l / m / 2
    return sum_cost + regularization

# backpropagation algorithm
def backpropagation(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels):
    m = x_data.shape[0]
    y_data = y_data.reshape((-1, 1))
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size*(input_layer_size + 1):].reshape((num_labels, -1))
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    # set delta3,delta2 to store some useful values
    delta3 = np.zeros((num_labels, 1))
    delta2 = np.zeros((hidden_layer_size, 1))
    for i in range(m):
        a1 = x_data[i,:].reshape((-1,1))
        label = y_data[i,:]
        # feedforward
        a1 = np.vstack((np.ones((1,1)),a1)) # add bais unit
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1,1)),a2)) 
        z3 = theta2 @ a2
        a3 = expit(z3) #hx

        for j in range(a3.shape[0]):
            if j+1 == label:
                flag = 1
            else:
                flag = 0
            delta3[j,0] = a3[j,0] - flag
        delta2 = np.multiply((theta2.T @ delta3)[1:], sigmoidGradient.sigmoid_gradient(z2))
        theta1_grad = theta1_grad + delta2 @ a1.T
        theta2_grad = theta2_grad + delta3 @ a2.T

    theta1_grad /= m
    theta2_grad /= m
    thetaGrad_vec = np.hstack((theta1_grad.reshape((1,-1)),theta2_grad.reshape((1,-1)))).flatten()
    return thetaGrad_vec

def backpropagation_reg(nn_params, input_layer_size, hidden_layer_size, x_data, y_data, num_labels, l):       
    m = x_data.shape[0]
    y_data = y_data.reshape((-1, 1))
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, -1))
    theta2 = nn_params[hidden_layer_size*(input_layer_size + 1):].reshape((num_labels, -1))
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    # set delta3,delta2 to store some useful values
    delta3 = np.zeros((num_labels, 1))
    delta2 = np.zeros((hidden_layer_size, 1))
    for i in range(m):
        a1 = x_data[i,:].reshape((-1,1))
        label = y_data[i,:]
        # feedforward
        a1 = np.vstack((np.ones((1,1)),a1)) # add bais unit
        z2 = theta1 @ a1
        a2 = expit(z2)
        a2 = np.vstack((np.ones((1,1)),a2)) 
        z3 = theta2 @ a2
        a3 = expit(z3) #hx

        for j in range(a3.shape[0]):
            if j+1 == label:
                flag = 1
            else:
                flag = 0
            delta3[j,0] = a3[j,0] - flag
        delta2 = np.multiply((theta2.T @ delta3)[1:], sigmoidGradient.sigmoid_gradient(z2))
        theta1_grad = theta1_grad + delta2 @ a1.T
        theta2_grad = theta2_grad + delta3 @ a2.T

    for i in range(theta1_grad.shape[0]):
        for j in range(theta1_grad.shape[1]):
            theta1_grad[i,j] /= m
            if j != 0:
                theta1_grad[i,j] += theta1[i,j] * l / m
    
    for i in range(theta2_grad.shape[0]):
        for j in range(theta2_grad.shape[1]):
            theta2_grad[i,j] /= m
            if j != 0:
                theta2_grad[i,j] += theta2[i,j] * l / m
            
    thetaGrad_vec = np.hstack((theta1_grad.reshape((1,-1)),theta2_grad.reshape((1,-1)))).flatten()
    return thetaGrad_vec