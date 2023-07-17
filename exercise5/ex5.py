from scipy.io import loadmat
import matplotlib.pyplot as plt 
import numpy as np
import linearRegCostFunction
import trainLinearReg
import learningCurve
import polyFeatures
import featureNormalize
import validationCurve
import random
# ================= loading and visualize dataset ===================================
dataset = loadmat("./machine-learning-ex5/ex5/ex5data1.mat")
x_train = dataset["X"]
y_train = dataset["y"]
x_val = dataset["Xval"]
y_val = dataset["yval"]
x_test = dataset["Xtest"]
y_test = dataset["ytest"]
m = x_train.shape[0]

plt.scatter(x_train, y_train,marker="*",color = "red")
plt.xlabel("Change in water level(x)")
plt.ylabel("Water flowing out of dam")
# plt.show()
# ================ Regularized Linear regression cost ===============
theta = np.array([1,1])
J = linearRegCostFunction.cost_reg(theta, x_train,y_train,lamda=1)
print("with theta = [1;1],l=1 cost is {}".format(J))
print("expected cost is 303.993192")
# =============== Regularized linear regression grad ================
grad = linearRegCostFunction.gradient_reg(theta, x_train, y_train, 1)
print("with theta = [1;1],l=1 grad is {}".format(grad))
print("expected grad is [-15.303016; 598.250744]")
# =============== train linear regression params ========================
# minimize cost with lambda = 0
initial_theta = np.zeros((x_train.shape[1] + 1,1))
final_theta, status = trainLinearReg.trainWithoutReg(initial_theta,x_train, y_train,lamda=0)
if status == 0:
    x_fit = np.insert(x_train, 0, 1, axis=1)
    y_fit = x_fit @ final_theta
    plt.plot(x_train,y_fit)
    plt.title("fitting curve when lamda is 0")
    plt.show()
# ================ learning curve for linear regression ===================
errors_train, errors_val = learningCurve.getError(x_train,y_train, x_val, y_val)
plt.figure()
plt.title("learning curve for linear regression")
plt.plot(np.arange(1, m+1), errors_val)
plt.plot(np.arange(1, m+1), errors_train)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("the size of trainDataSet")
plt.ylabel("errors")
plt.show()
# ================ feature mapping for polynomial regression ===============
# select model
# p = [2,3,4,5,6,7,8]

# error_train = np.zeros((len(p),1))
# error_val = np.zeros((len(p),1))
# for i in range(len(p)):
#     x_poly = polyFeatures.mapFeatures(x_train,p[i]) # m * item
#     x_poly_val = polyFeatures.mapFeatures(x_val,p[i])
#     initial_theta = np.zeros((x_poly.shape[1] + 1,1))
#     final_theta,_ = trainLinearReg.trainWithoutReg(initial_theta,x_poly,y_train,lamda = 0)
#     e_train = linearRegCostFunction.cost_reg(final_theta,x_poly,y_train,lamda=0)
#     e_val = linearRegCostFunction.cost_reg(final_theta,x_poly_val,y_val,lamda=0)
#     error_train[i, 0] = e_train
#     error_val[i,0] = e_val
# plt.figure()
# plt.plot(p,error_val.flatten())
# plt.plot(p, error_train.flatten())
# plt.legend(("error_val", "error_train"))
# plt.xlabel("the size of d")
# plt.ylabel("errors")
# plt.show()

# as we can see, d = 6 is  overfit, so we choose d = 5
P = 5
# map and featurenomalize
x_poly = polyFeatures.mapFeatures(x_train,P)
x_poly_train,mu_train,std_train = featureNormalize.feature_scaling(x_poly)
x_poly_val = polyFeatures.mapFeatures(x_val,P)
x_poly_val = (x_poly_val - mu_train) / std_train
initial_theta = np.zeros((x_poly_train.shape[1] + 1,1))
after_poly_theta,_ = trainLinearReg.trainWithoutReg(initial_theta,x_poly_train,y_train,0)
# plot fitting curve
plt.figure()
plt.title("the fitting curve when degree equals 5")
plt.scatter(x_train, y_train)
# x_poly_train = np.insert(x_poly_train,0,1,axis=1)
# plt.plot(x_train, x_poly_train@after_poly_theta)
x_plot = np.linspace(-55, 55, 50).reshape((-1, 1))
x_plot_nomal = polyFeatures.mapFeatures(x_plot, P)
x_plot_nomal = (x_plot_nomal - mu_train) / std_train
x_plot_nomal = np.insert(x_plot_nomal, 0, 1, axis=1)
plt.plot(x_plot,x_plot_nomal @after_poly_theta)
plt.xlabel("Change in water level")
plt.ylabel("water flowing out of dam")
plt.show()
# ======================  learning curve for polynomial regression =======================
plt.figure()
plt.title("learning curve for polynomial regression when degree equals 5")
train_errors, cv_errors = learningCurve.getError(x_poly_train, y_train, x_poly_val, y_val)
plt.plot(np.arange(1, m+1), cv_errors)
plt.plot(np.arange(1, m+1), train_errors)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("the size of trainDataSet")
plt.ylabel("errors")
plt.show()
# ====================== select the best lambda ============================================
# l = 1
lamda = 1
final_theta_l1,_ = trainLinearReg.trainWithoutReg(initial_theta,x_poly_train,y_train,lamda)
# plot fitting curve
plt.figure()
plt.title("fitting curve when lamda is 1")
plt.scatter(x_train, y_train)
# x_poly_train = np.insert(x_poly_train,0,1,axis=1)
# plt.plot(x_train, x_poly_train@after_poly_theta)
x_plot = np.linspace(-55, 55, 50).reshape((-1, 1))
x_plot_nomal = polyFeatures.mapFeatures(x_plot, P)
x_plot_nomal = (x_plot_nomal - mu_train) / std_train
x_plot_nomal = np.insert(x_plot_nomal, 0, 1, axis=1)
plt.plot(x_plot,x_plot_nomal @final_theta_l1)
plt.xlabel("Change in water level")
plt.ylabel("water flowing out of dam")
plt.show()
# plot learning curve 
plt.figure()
plt.title("learning curve for polynomial regression  when lamda is 1")
train_errors, cv_errors = learningCurve.getError(x_poly_train, y_train, x_poly_val, y_val)
plt.plot(np.arange(1, m+1), cv_errors)
plt.plot(np.arange(1, m+1), train_errors)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("the size of trainDataSet")
plt.ylabel("errors")
plt.show()

# =========== lamda = 10 ==================
lamda = 10
final_theta_l1,_ = trainLinearReg.trainWithoutReg(initial_theta,x_poly_train,y_train,lamda)
# plot fitting curve
plt.figure()
plt.title("fitting curve when lamda is 10")
plt.scatter(x_train, y_train)
# x_poly_train = np.insert(x_poly_train,0,1,axis=1)
# plt.plot(x_train, x_poly_train@after_poly_theta)
x_plot = np.linspace(-55, 55, 50).reshape((-1, 1))
x_plot_nomal = polyFeatures.mapFeatures(x_plot, P)
x_plot_nomal = (x_plot_nomal - mu_train) / std_train
x_plot_nomal = np.insert(x_plot_nomal, 0, 1, axis=1)
plt.plot(x_plot,x_plot_nomal @final_theta_l1)
plt.xlabel("Change in water level")
plt.ylabel("water flowing out of dam")
plt.show()
# plot learning curve 
plt.figure()
plt.title("learning curve for polynomial regression  when lamda is 10")
train_errors, cv_errors = learningCurve.getError(x_poly_train, y_train, x_poly_val, y_val)
plt.plot(np.arange(1, m+1), cv_errors)
plt.plot(np.arange(1, m+1), train_errors)
plt.legend(("errors_val", "errors_train"))
plt.xlabel("the size of trainDataSet")
plt.ylabel("errors")
plt.show()
# =======================Selecting λ using a cross validation set========================
lamdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
train_errors, cv_errors = validationCurve.autoSelect(x_poly_train,y_train,x_poly_val,y_val,lamdas)
plt.figure()
plt.title("training errors and cv errors in different values of lamda")
plt.plot(lamdas,cv_errors.flatten())
plt.plot(lamdas,train_errors.flatten())
plt.legend((" CVerrors", "Train errors"))
plt.xlabel("lamda")
plt.ylabel("errors")
plt.show()
# in my picture, best lamda around 1 
best_selected_theta, _ = trainLinearReg.trainWithoutReg(initial_theta, x_poly_train,y_train,lamda=1)
# map test
x_test_poly = polyFeatures.mapFeatures(x_test,P)
x_test_poly = (x_test_poly - mu_train) / std_train
J_ = linearRegCostFunction.cost_reg(best_selected_theta, x_test_poly,y_test,lamda=1)
print("test error of {} for λ = 1".format(J_))
# ================= Plotting learning curves with randomly selected examples =====================
lamda = 0.01
total_samples = min(len(x_train), len(x_val))
train_indices = list(range(len(x_train)))
cv_indices = list(range(len(x_val)))

errors_train_samples = np.zeros((total_samples,1))
errors_cv_samples = np.zeros((total_samples,1))
for i in range(1,total_samples):
    x_train_random_indices = random.sample(train_indices, i + 1)
    x_val_random_indices = random.sample(cv_indices, i + 1)
    x_train_random_sample = [x_train[i] for i in x_train_random_indices]
    y_train_random_sample = [y_train[i] for i in x_train_random_indices]
    x_val_random_sample = [x_val[i] for i in x_val_random_indices]
    y_val_random_sample = [y_val[i] for i in x_val_random_indices]

    x_train_random_sample = np.array(x_train_random_sample)
    x_val_random_sample = np.array(x_val_random_sample)
    y_train_random_sample = np.array(y_train_random_sample)
    y_val_random_sample = np.array(y_val_random_sample)
    #map feature
    x_train_random_sample_poly = polyFeatures.mapFeatures(x_train_random_sample, P)
    x_val_random_sample_poly = polyFeatures.mapFeatures(x_val_random_sample, P)
    x_train_random_sample_poly, mu_sample, std_sample = featureNormalize.feature_scaling(x_train_random_sample_poly)
    x_val_random_sample_poly = (x_val_random_sample_poly - mu_sample) / std_sample

    initial_theta = np.zeros((x_train_random_sample_poly.shape[1] + 1,1))
    final_theta,_ = trainLinearReg.trainWithoutReg(initial_theta,x_train_random_sample_poly,y_train_random_sample,lamda=lamda)
    errors_train_sample, errors_val_sample = learningCurve.getError(x_train_random_sample_poly,y_train_random_sample, x_val_random_sample_poly, y_val_random_sample)
    errors_train_samples[i, 0] = np.mean(errors_train_sample)
    errors_cv_samples[i, 0] = np.mean(errors_val_sample)

print(errors_train_samples)
print(errors_cv_samples)
plt.figure()
plt.title("Plotting learning curves with randomly selected examples")

plt.plot(np.arange(1, total_samples + 1), errors_train_samples.flatten())
plt.plot(np.arange(1, total_samples + 1), errors_cv_samples.flatten())
plt.legend(("Train", "Cross Validation"))
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.show()