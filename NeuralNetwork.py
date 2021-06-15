import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utilis import plot_decision_boundary, sigmoid, load_plain_dataset

np.random.seed(1)

X, Y = load_plain_dataset()
'''plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()'''

'''clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")'''


#------defining Layer sizes ------------
def layer_sizes(X, Y) :
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


'''x_ls = np.random.randn(5,3)
y_ls = np.random.randn(2,3)
(n_x, n_h, n_y) = layer_sizes(x_ls, y_ls)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))'''


#--------------Initializinfg the parameters w1, w2, b1, b2 -------------
def initialize_params(n_x, n_h, n_y):
    np.random.seed(2)
    w1 =  np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert(w1.shape == (n_h, n_x))
    assert(w2.shape == (n_y, n_h))
    assert(b1.shape == (n_h, 1))
    assert(b2.shape == (n_y, 1))

    params = {"w1" : w1, "w2" : w2, "b1" : b1, "b2" : b2}
    return params

'''n_x, n_h, n_y = 2, 4, 1
params = initialize_params(n_x, n_h, n_y)
print("W1 = " + str(params["w1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["w2"]))
print("b2 = " + str(params["b2"]))'''

#---------Forward Propagations -------------

def forward_prop(X, params) :
    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    assert(a2.shape == (1, X.shape[1]))

    cache = {"z1" : z1, "z2" : z2, "a1" : a1, "a2" : a2}

    return a2, cache

#------------Computing Cost -----------------

def compute_cost(a2, y, params) :
    m = y.shape[1]
    loss = np.dot(np.log(a2), (y.T)) + np.dot(np.log(1-a2), ((1-y).T))
    cost = -loss/m
    cost = float(np.squeeze(cost))
    assert(isinstance(cost, float))
    return cost

#-------------Backward Propagation ------------------

def backward_prop(params, cache, X, y) :
    m = y.shape[1]

    a1 = cache["a1"]
    a2 = cache["a2"]

    w1 = params["w1"]
    w2 = params["w2"]

    dz2 = a2 - y
    dw2 = (1/m)*np.dot(dz2, a1.T)
    db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
    #--------------you can compute $g^{[1]'}(Z^{[1]})$ using (1 - np.power(A1, 2)).--------------
    dz1 = np.dot(w2.T, dz2)*(1-np.power(a1, 2))
    dw1 = (1/m)*np.dot(dz1, X.T)
    db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)

    grads = {"dw1" : dw1, "dw2" : dw2, "db1" : db1, "db2" : db2}

    return grads

#-------------------Update rule------------------

def update(grads, params, alpha) :

    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]

    dw1 = grads["dw1"]
    dw2 = grads["dw2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    w1 -= alpha*dw1
    w2 -= alpha*dw2
    b1 -= alpha*db1
    b2 -= alpha*db2

    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_steps, alpha):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    params = initialize_params(n_x, n_h, n_y)
    

    for i in range(num_steps) :
        a2, cache = forward_prop(X, params)
        cost = compute_cost(a2, Y, params)
        grads = backward_prop(params, cache, X, Y)
        params = update(grads, params, alpha)

        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return params












    





    
    
