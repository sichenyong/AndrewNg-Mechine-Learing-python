import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cofiCostFunc
import checkCostFunction
import scipy.optimize as op

# =============== Part 1: Loading movie ratings dataset ================
print('Loading movie ratings dataset.\n')
data =loadmat("./machine-learning-ex8/ex8/ex8_movies.mat")
Y = data["Y"]
R = data["R"]
print('Loading movie params dataset.\n')
data1 =loadmat("./machine-learning-ex8/ex8/ex8_movieParams.mat")
X = data1["X"] # 1682 * 10
Theta = data1["Theta"] # 943 * 10
nm = data1["num_movies"] # 1682
nu = data1["num_users"] # 943
nf = data1["num_features"] # 10
# bool index
print('Average rating for movie 1 (Toy Story):{:.2f} / 5\n'.format(np.mean(Y[0,R[0,:]])))

# We can "visualize" the ratings matrix by plotting it with imshow
# plt.figure(num=1)
# plt.imshow(Y, cmap='hot', aspect='auto')
# plt.colorbar()
# plt.xlabel('Movies')
# plt.ylabel('Users')
# plt.show()
# ============ Part 2: Collaborative Filtering Cost Function ===========
num_users = 4
num_movies = 5
num_features = 3
X_some = X[0:num_movies, 0:num_features]
Theta_some = Theta[0:num_users, 0:num_features]
Y_some = Y[0:num_movies, 0:num_users]
R_some = R[0:num_movies, 0:num_users]
# evaluate cost function
params = np.hstack((X_some.flatten(), Theta_some.flatten()))
J = cofiCostFunc.cost(params,Y_some,R_some,num_users,num_movies, num_features)
print('Cost at loaded parameters: {:.2f}\n'.format(J))
print('expected value should be about 22.22')
# check with no reg
grad = cofiCostFunc.gradient(params,Y_some,R_some,num_users,num_movies, num_features)
grad_numerical = checkCostFunction.compute_grad_numerically(params,Y_some,R_some,num_users,num_movies, num_features)
print(grad)
print(grad_numerical)
print('\n')

# check with reg
cost = cofiCostFunc.cost(params,Y_some,R_some,num_users,num_movies, num_features,1.5)
print("cost is {:.2f}, expected cost is 31.34".format(cost))
grad = cofiCostFunc.gradient(params,Y_some,R_some,num_users,num_movies, num_features,1.5)
grad_numerical = checkCostFunction.compute_grad_numerically(params,Y_some,R_some,num_users,num_movies, num_features,1.5)
print(grad)
print(grad_numerical)
print('\n')
#  ============== Part 6: Entering ratings for a new user ===============
movie_list = []
with open("./machine-learning-ex8/ex8/movie_ids.txt",'r',encoding='latin-1') as file:
    lines = file.readlines()
    for line in lines:
        strings = line.strip().split(' ')
        movie_list.append(' '.join(strings[1:]))
# reproduce my ratings
ratings = np.zeros(1682)
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

# i as user0
Y = np.insert(Y, 0, ratings, axis=1)
R = np.insert(R, 0, ratings != 0, axis=1)  # type casting
# some params
n_features = 50
n_movie, n_user = Y.shape
l = 10
# 随机初始化
X = np.random.standard_normal((n_movie, n_features))
Theta = np.random.standard_normal((n_user, n_features))
params = np.hstack((X.flatten(), Theta.flatten()))
# normalized ratings
Y_mean = np.reshape(np.mean(Y, 1), (Y.shape[0], 1))
normalized_Y = Y - Y_mean
# learing x theta
print('\nTraining collaborative filtering...\n')
res = op.minimize(fun=cofiCostFunc.cost, x0=params, args=(normalized_Y,R,n_user, n_movie, n_features, l)
                  ,method='TNC', jac=cofiCostFunc.gradient)
print(res)
params = res.x
X_final = params[0:n_movie*n_features].reshape((n_movie,-1))
Theta_final = params[n_movie*n_features:].reshape((n_user, -1))
print('Recommender system learning completed.\n')
# ================== Part 8: Recommendation for you ====================
p = X_final @ Theta_final.T
my_predictions = p[:,0].reshape((-1, 1)) + Y_mean

# 推荐个数
n = 10
idx = np.argsort(-my_predictions.flatten())[0:n]
for m in idx:
    # show top 10
    print(movie_list[m])


