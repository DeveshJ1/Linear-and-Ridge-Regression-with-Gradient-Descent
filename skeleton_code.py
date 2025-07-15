import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size(num_instances, num_features)
        test - test set, a 2D numpy array of size(num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    #find the min and max of training set in order to evaluate constant features
    min_val= np.min(train, axis=0)
    max_val= np.max(train, axis=0)
    #ignore constant features as said in the instructions
    constant_features= (max_val!=min_val)
    #normalize features that are not constant
    train_normalized=(train[:,constant_features]-min_val[constant_features])/(max_val[constant_features]-min_val[constant_features])
    test_normalized=(test[:,constant_features]-min_val[constant_features])/(max_val[constant_features]-min_val[constant_features])
    return train_normalized,test_normalized

#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D array of size(num_features)

    Returns:
        loss - the average square loss, scalar
    """
    #TODO
    #matrix multiplication to get vector of predicted y's
    predicted_y= X @ theta
    #calculate the average square loss using numpy's mean function 
    loss=np.mean((predicted_y-y)**2)
    return loss

#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss(as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO
    #get the number of instances
    m=X.shape[0]
    #calculate the gradient according to formula derived in question 3
    grad=(2/m) * X.T @ (X @ theta -y)
    #return the gradient
    return grad

#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
(e_1 =(1,0,0,...,0), e_2 =(0,1,0,...,0), ..., e_d =(0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
(J(theta + epsilon * e_i) - J(theta - epsilon * e_i)) /(2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    #get the loss of the true gradient
    true_gradient = compute_square_loss_gradient(X, y, theta) 
    num_features = theta.shape[0]
    #Initialize the gradient we approximate
    approx_grad = np.zeros(num_features) 
    #TODO
    #going through each feature for each theta we calculate the square loss and store it 
    for i in range (num_features):
        theta_add=np.copy(theta)
        theta_subtract=np.copy(theta)
        theta_add[i]+=epsilon
        theta_subtract[i]-=epsilon
        #according to formula we derived
        approx_grad[i]=(compute_square_loss(X,y,theta_add)-compute_square_loss(X,y,theta_subtract))/ (2*epsilon)
        #get the error respective to true gradient
    error=np.linalg.norm(true_gradient-approx_grad)
    return error<=tolerance

#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, 
                             epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO
    #get the gradient
    true_gradient=gradient_func(X,y, theta)
    feature_amt=theta.shape[0]
    approx_grad=np.zeros(feature_amt)
    #pretty much similar to what we did above
    for i in range (feature_amt):
        theta_add=np.copy(theta)
        theta_subtract=np.copy(theta)
        theta_add[i]+=epsilon
        theta_subtract[i]-=epsilon
        approx_grad[i]=(objective_func(X,y,theta_add)-objective_func(X,y,theta_subtract))/ (2*epsilon)
    error=np.linalg.norm(true_gradient-approx_grad)
    return error<=tolerance

#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    #TODO
    #go through each step  
    for i in range(num_step):
        #update theta and loss for each step
        theta_hist[i]=theta
        loss_hist[i]=compute_square_loss(X,y,theta)
        #calculate the gradient
        gradient=compute_square_loss_gradient(X,y,theta)
        #check the gradient 
        if grad_check:
            if not grad_checker(X,y,theta):
                sys.exit("Incorrect Gradient!")
        #update theta for next step
        theta-=alpha * gradient
    #final theta and lost computations 
    theta_hist[num_step]=theta
    loss_hist[num_step]=compute_square_loss(X,y,theta)
    return theta_hist, loss_hist
#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO
    m=X.shape[0]
    #calculate gradient vector according to formula derived in question 11
    grad=(2/m) * X.T @ (X @ theta - y) + 2 *lambda_reg * theta
    return grad
#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    #TODO
    #pretty much same as what we did before for batch_gradient_descent
    #but now we have a regularized term to consider
    for i in range(num_step):
        theta_hist[i]=theta
        loss_hist[i]=compute_square_loss(X,y,theta)
        gradient=compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        theta-= alpha *gradient
    theta_hist[num_step]=theta
    loss_hist[num_step]=compute_square_loss(X,y,theta)
    return theta_hist, loss_hist

def load_data():
    #Loading the dataset
    print('loading the dataset')
    #i had to change the path for my computer which is why it is different 
    #however i included the path that was given below 
   # df = pd.read_csv('Downloads/hw1/ridge_regression_dataset.csv', delimiter=',')
    df = pd.read_csv('refs/hw2_2020/ridge_regression_dataset.csv', delimiter=',')

    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
#for the plot questions i encountered errors when alpha was too large
#or when labmda was to large due to an overflow and I read on campuswire
#that this was expected so I wasn't sure if it was a mistake or not
'''
#Question 9 code 
fig = plt.figure(figsize=(8,6))
sp = fig.add_subplot(111)
for a in [0.5,0.1,0.05, 0.01,0.005,0.001]:
    theta_hist1, loss_hist1 = batch_grad_descent(X_train, y_train, alpha=a)
    plt.plot(loss_hist1, label='step size = %r' %a)
sp.set_xlabel('Steps')
sp.set_ylabel('Average square loss')
sp.set_yscale('log')
plt.title('Average Square Loss of Iterations for Different Step Sizes of BGD')
plt.legend(loc='best')
plt.show()
'''
'''
#Question 10 code:

optimal_step_size = 0.05

theta_hist, train_loss_hist = batch_grad_descent(X_train, y_train, alpha=optimal_step_size, num_step=num_step)

# Compute test loss history
_, test_loss_hist = batch_grad_descent(X_test, y_test, alpha=optimal_step_size)

# Plot the train and test loss histories
plt.plot(range(num_step + 1), train_loss_hist, label="Train Loss")
plt.plot(range(num_step + 1), test_loss_hist, label="Test Loss")
plt.xlabel("Iterations")
plt.ylabel("Average Square Loss")
plt.title("Train vs Test Loss Over Iterations")
plt.legend()
plt.show()
'''
'''
#Question 14 code

lambdas = [1e-7, 1e-5, 1e-3, 1e-1,1,10,100]
alpha = 0.05

for lambda_reg in lambdas:
    theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, alpha, lambda_reg)
    _, test_loss_hist = regularized_grad_descent(X_test, y_test, alpha, lambda_reg)

    # Plot the loss histories
    plt.plot(range(1001), loss_hist, label=f"Train Loss, λ={lambda_reg}")
    plt.plot(range(1001), test_loss_hist, label=f"Test Loss, λ={lambda_reg}")
    


plt.xlabel("Iterations")
plt.ylabel("Average Square Loss")
plt.title("Train and Test Loss for Different λ Values")
plt.legend()
plt.show()
'''
#Question 15 code
'''
train_loss = []
test_loss = []
for l in np.logspace(-7, -1, num=30):
    theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, lambda_reg=l)
    train_loss.append(loss_hist[-1])
    test_loss.append(compute_square_loss(X_test, y_test, theta_hist[-1]))

c = np.linspace(-7, -1, num=30)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
plt.plot(c, train_loss, label='training loss')
plt.plot(c, test_loss, label='validation loss')

ax.set_xlabel('$\log_{10}\lambda$')
ax.set_ylabel('Average square loss')
plt.title('Average square loss as a function of regularization coefficients')
plt.legend(loc='best')
plt.show()
'''