# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:56:31.521739Z","iopub.execute_input":"2021-07-06T12:56:31.522164Z","iopub.status.idle":"2021-07-06T12:56:31.535364Z","shell.execute_reply.started":"2021-07-06T12:56:31.522132Z","shell.execute_reply":"2021-07-06T12:56:31.534324Z"}}
#DIGIT RECOGNIZER
#THIS IS THE SOLUTION OF SHIKHA BHAT (user: shikha1608) FROM kaggle.com competitions solution
#I am rewriting SHIKHA's code to practice and learn.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import math
import sys
from scipy import optimize
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:56:31.746611Z","iopub.execute_input":"2021-07-06T12:56:31.746988Z","iopub.status.idle":"2021-07-06T12:56:34.301529Z","shell.execute_reply.started":"2021-07-06T12:56:31.746954Z","shell.execute_reply":"2021-07-06T12:56:34.300569Z"}}
csv = '../input/digit-recognizer/train.csv'
data = pd.read_csv(csv)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:50:16.836366Z","iopub.execute_input":"2021-07-06T12:50:16.836637Z","iopub.status.idle":"2021-07-06T12:50:16.863028Z","shell.execute_reply.started":"2021-07-06T12:50:16.836612Z","shell.execute_reply":"2021-07-06T12:50:16.862107Z"}}
data

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:50:36.742529Z","iopub.execute_input":"2021-07-06T12:50:36.742911Z","iopub.status.idle":"2021-07-06T12:50:37.562713Z","shell.execute_reply.started":"2021-07-06T12:50:36.742879Z","shell.execute_reply":"2021-07-06T12:50:37.561960Z"}}
display = np.matrix(data)
output = display[:,0]
display = np.delete(display,0,1)
m = output.size
rand_indices = np.random.choice(m, 5, replace = False)

for i in rand_indices:
    img = display[i].reshape(28,28)
    plt.figure()
    plt.imshow(img, cmap = "gray")


# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:50:56.785577Z","iopub.execute_input":"2021-07-06T12:50:56.785934Z","iopub.status.idle":"2021-07-06T12:50:56.791015Z","shell.execute_reply.started":"2021-07-06T12:50:56.785904Z","shell.execute_reply":"2021-07-06T12:50:56.790159Z"}}
display.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:54:37.900591Z","iopub.execute_input":"2021-07-06T12:54:37.901000Z","iopub.status.idle":"2021-07-06T12:54:37.905680Z","shell.execute_reply.started":"2021-07-06T12:54:37.900969Z","shell.execute_reply":"2021-07-06T12:54:37.904711Z"}}
input_layer_size = 784
hidden_layer_size = 50
num_labels = 10


# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T12:59:53.431585Z","iopub.execute_input":"2021-07-06T12:59:53.431928Z","iopub.status.idle":"2021-07-06T12:59:53.435955Z","shell.execute_reply.started":"2021-07-06T12:59:53.431901Z","shell.execute_reply":"2021-07-06T12:59:53.435105Z"}}
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T13:06:30.417504Z","iopub.execute_input":"2021-07-06T13:06:30.417883Z","iopub.status.idle":"2021-07-06T13:06:30.422022Z","shell.execute_reply.started":"2021-07-06T13:06:30.417852Z","shell.execute_reply":"2021-07-06T13:06:30.421276Z"}}
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    s = sigmoid(z)
    g = np.multiply(s, (1-s))
    
    return g

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T13:12:48.385351Z","iopub.execute_input":"2021-07-06T13:12:48.385879Z","iopub.status.idle":"2021-07-06T13:12:48.390672Z","shell.execute_reply.started":"2021-07-06T13:12:48.385833Z","shell.execute_reply":"2021-07-06T13:12:48.389912Z"}}
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    
    #adding a bias layer
    W = np.zeros((L_out, 1 + L_in))
    
    #initializing weights
    W = np.random.rand(L_out, 1 + L_in)*2*epsilon_init - epsilon_init
    
    print(W.shape)
    
    return W

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T13:18:20.890511Z","iopub.execute_input":"2021-07-06T13:18:20.891052Z","iopub.status.idle":"2021-07-06T13:18:20.898080Z","shell.execute_reply.started":"2021-07-06T13:18:20.891010Z","shell.execute_reply":"2021-07-06T13:18:20.897041Z"}}
print("Initializing Neural Network Parameters ...")


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis = 0)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:05:28.507131Z","iopub.execute_input":"2021-07-06T14:05:28.507511Z","iopub.status.idle":"2021-07-06T14:05:28.523209Z","shell.execute_reply.started":"2021-07-06T14:05:28.507476Z","shell.execute_reply":"2021-07-06T14:05:28.522250Z"}}
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, X, y, num_labels, lamb = 0.0):
    """
    Implements the neural network regularized cost function and gradient for a two layer neural 
    network which performs classification. 
    Feedforward the neural network and return the cost in the variable J. Implement the backpropagation 
    algorithm to compute the gradients Theta1_grad and Theta2_grad. Implement regularization with the 
    cost function and gradients.
    
    """
    
    #number of examples in set
    m = y.size
    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                       (hidden_layer_size, (input_layer_size+1)))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    X = np.concatenate([np.ones((m,1)), X], axis = 1)
    
    #creating necessary yk array
    #if in 4th example, y = 9, then set 9th row, 4th column of yk as 1
    #similarly do for all examples
    yk = np.zeros((num_labels, m))
    for i in range (m):
        yk[ y[i], i ] = 1
        
    '''
    Forward Propagation
    '''
    
    #first layer
    z2 = Theta1@X.T
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((1,m)), a2], axis=0)
    
    #second layer
    z3 = Theta2@a2
    a3 = sigmoid(z3)
    
    #to not account for bias in regularization
    t1 = Theta1[:, 1:]
    t2 = Theta2[:, 1:]
    
    #regularized cost
    J = 1/m * np.sum(np.sum(-np.multiply(yk,np.log(a3)) - np.multiply((1-yk),
            np.log(1-a3)))) + lamb/(2*m) * (np.sum(np.sum(t1**2))+np.sum(np.sum(t2**2)))
    
    '''
    Backward propagation
    '''
    
    d3 = a3 - yk
    
    z2 = np.concatenate([np.ones((1,m)), z2], axis=0)
    d2 = np.multiply(Theta2.T@d3, sigmoidGradient(z2))
    
    Theta2_grad = Theta2_grad + d3@a2.T
    Theta1_grad = Theta1_grad + d2[1:]@X
    
    
    Theta1_grad[:,0] = np.divide(Theta1_grad[:,0],m)
    Theta1_grad[:,1:] = np.divide(Theta1_grad[:,1:],m) + (lamb/m)*Theta1[:,1:]
    
    Theta2_grad[:,0] = np.divide(Theta2_grad[:,0],m)
    Theta2_grad[:,1:] = np.divide(Theta2_grad[:,1:],m)+ (lamb/m)*Theta2[:,1:]
    
    r1 = Theta1_grad.ravel()
    r2 = Theta2_grad.ravel()
    
    grad = np.concatenate([r1.T,r2.T])
    
    return J, grad
    

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:05:54.849611Z","iopub.execute_input":"2021-07-06T14:05:54.849981Z","iopub.status.idle":"2021-07-06T14:12:09.636392Z","shell.execute_reply.started":"2021-07-06T14:05:54.849951Z","shell.execute_reply":"2021-07-06T14:12:09.635422Z"}}
options= {'maxiter': 400}

lambda_ = 9

costFunction = lambda p: nnCostFunction(p, 
                                        input_layer_size, hidden_layer_size, display, output, num_labels, lambda_)

res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

nn_params = res.x
        
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:12:09.639063Z","iopub.execute_input":"2021-07-06T14:12:09.639477Z","iopub.status.idle":"2021-07-06T14:12:10.071358Z","shell.execute_reply.started":"2021-07-06T14:12:09.639440Z","shell.execute_reply":"2021-07-06T14:12:10.070334Z"}}
def predict(Theta1, Theta2, X):
    
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    
    return p


pred = predict(Theta1, Theta2, display)
print('Training Set Accuracy: %f' % (np.mean(pred == output) * 100))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:12:10.072881Z","iopub.execute_input":"2021-07-06T14:12:10.073464Z","iopub.status.idle":"2021-07-06T14:12:12.390197Z","shell.execute_reply.started":"2021-07-06T14:12:10.073417Z","shell.execute_reply":"2021-07-06T14:12:12.389155Z"}}
csv = '../input/digit-recognizer/test.csv'
test = pd.read_csv(csv)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:12:12.391313Z","iopub.execute_input":"2021-07-06T14:12:12.391568Z","iopub.status.idle":"2021-07-06T14:12:12.751806Z","shell.execute_reply.started":"2021-07-06T14:12:12.391543Z","shell.execute_reply":"2021-07-06T14:12:12.750490Z"}}
X = np.matrix(test)
m = X.shape[0]
# X.shape
pred = predict(Theta1, Theta2, X)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-06T14:12:12.754800Z","iopub.execute_input":"2021-07-06T14:12:12.755254Z","iopub.status.idle":"2021-07-06T14:12:12.993451Z","shell.execute_reply.started":"2021-07-06T14:12:12.755209Z","shell.execute_reply":"2021-07-06T14:12:12.992507Z"}}
ans = [[0,0] for i in range(m)]
for i in range(m):
    ans[i][0] = i+1
    ans[i][1] = int(pred[i])

df = pd.DataFrame(ans, columns = ['ImageId', 'Label'])
df.head(20)

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
