# %%
import math , copy
import numpy as np
import matplotlib.pyplot as plt

# %%
def load_house_data():
    data = np.loadtxt(r"D:\AndrewNgOptionalLabs\supervised\houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

# %%
# load dataset
X_train , y_train = load_house_data()
X_features = ['size(sqft)' , 'bedrooms', 'floors' , 'age']

# %%
# compute cost
# mean squared error for our predictions
def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(w , X[i]) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost /= (2 * m)
    return cost

# %%
# gradient descent
def compute_gradient(X,y,w,b):
    m , n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range (m):
        # err =  f_wb -> (w * xi + b ) - y
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw , dj_db

# %%
def gradient_descent(X,y,w_in,b_in,cost_fun,gradient_fun,alpha,num):
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    for i in range(num):
        # Calculate the gradient and update the parameters
        dj_dw , dj_db = gradient_fun(X,y,w,b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # save the cost J 
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_fun(X, y, w, b))
         # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# %%
#set alpha to 9.9e-7
m,n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0
wfinal , bfinal , J_hist = gradient_descent(X_train, y_train,initial_w , initial_b ,compute_cost,compute_gradient, 9.9e-7 , 10)
wfinal

# %%
#set alpha to 9e-7
m,n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0
wfinal , bfinal , J_hist = gradient_descent(X_train, y_train,initial_w , initial_b ,compute_cost,compute_gradient, 9e-7 , 10)
wfinal

# %%
# alpha = 1e-7
wfinal , bfinal , J_hist = gradient_descent(X_train, y_train,initial_w , initial_b ,compute_cost,compute_gradient, 1e-7 , 10)
wfinal

# %%
# feature scaling

# %%
def zscore_normalize_features(X):
    # finding mean 
    mu = np.mean(X,axis=0)
    # find std dev of each column
    sigma = np.std(X,axis=0)
    # zscore
    X_norm = (X - mu) / sigma

    return X_norm , mu , sigma

# %%
#check our work
# !pip install scikit-learn

# %%
from sklearn.preprocessing import scale
scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)

# %%
X_norm , X_mu , X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# %%
w_norm, b_norm, hist = gradient_descent(X_norm, y_train,initial_w , initial_b ,compute_cost,compute_gradient,1.0e-1,1000)

# %%
# prediction using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i] , w_norm) + b_norm

# %%
# plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp, label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# %% [markdown]
# Prediction The point of generating our model is to use it to predict housing prices that are not in the data set. Let's predict the price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old. Recall, that you must normalize the data with the mean and standard deviation derived when the training data was normalized.

# %%
# prediction out of the data set
X_house = np.array([1200,3,1,40])
X_house_norm = (X_house - X_mu) / X_sigma
print(X_house_norm)
X_house_predict = np.dot(X_house_norm , w_norm ) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${X_house_predict*1000:0.0f}")

# %%



