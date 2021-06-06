import numpy as np
import pandas as pd
from numpy.core.fromnumeric import transpose   


def SegmoidFunc(theta_vector, X_matrix):        # The implementation of the segmoid function
    h = np.dot(X_matrix, theta_vector)
    return 1 / (1 + np.e**(-1 * h))


def CostFunc(theta_vector, X_matrix, Y_vector):
    Z = np.round(SegmoidFunc(theta_vector, X_matrix), 10)
    print('Z = ', Z)
    sum = np.dot(Y_vector.T, np.log(Z)) + np.dot((1 - Y_vector).T, np.log(1 - Z))
    m = Y_vector.shape
    m = m[0]
    return -1 * (sum / m) 


def GradientDescentFunc(theta_vector, alpha, X_matrix, Y_vector, lamda):
    Z = SegmoidFunc(theta_vector, X_matrix)
    sum = np.dot(X_matrix.T, (Z - Y_vector))
    reguralization_term = 0 # The reguralization term will be zero 
                            # becauase we already use a straight line to clasify the data 
                            # so we have not an overfitting to be reguralized
    # reguralization_term = (lamda / (2 * Y_vector.size)) * (np.sum(theta_vector@theta_vector, axis=0))
    m = Y_vector.shape
    m = m[0]
    return (alpha / m) * sum + reguralization_term
        

def Train(file_name, iterations, alpha, lamda):
    
    Data_Frame = pd.read_csv(file_name)
    
    TrainingSet = np.array(Data_Frame)
    
    # features = X
    features_matrix = TrainingSet[:,0:-1]
    # values = Y
    values = TrainingSet[:,-1].reshape(212,1)

    theta_vector = np.ones(features_matrix[0].size).reshape(14,1)
    error = 0

    # iterating to change theta
    for i in range(iterations):
        try:
            if i < iterations - 1:
                if i == 10:
                    break
                error = CostFunc(theta_vector, features_matrix, values)
                print('Iteration #{}: Error = {} , Thetas = {}'.format(i + 1, error, theta_vector))
                theta_vector -= GradientDescentFunc(theta_vector, alpha, features_matrix, values, lamda)
            else:
                print('Final Iteration #{}: Final Error = {} , Final Thetas = {}'.format(i + 1, error, theta_vector))  
        except:
            print("Error: Logistic regression process, Error Explain: I can't learn!")
            break
        
    try:
        pd.DataFrame(theta_vector).to_csv("./results/results.csv", index=False, header=False)
    except:
        print("Error: DescalingFeatures process, Error Explain: Dividing on zero!")
    
    print(' ' + '-' * 29)
    print('| Training done successfully! |')
    print(' ' + '-' * 29)