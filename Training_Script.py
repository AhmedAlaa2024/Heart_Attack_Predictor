import numpy as np
import pandas as pd
from numpy.core.fromnumeric import transpose   


def SegmoidFunc(theta_vector, X_matrix):        # The implementation of the segmoid function
    
    return 1 / (1 + np.e**(-1 * np.sum(theta_vector * X_matrix,axis=1)))


def CostFunc(theta_vector, X_matrix, Y_vector):
    Z = SegmoidFunc(theta_vector, X_matrix)
    sum = np.nansum(Y_vector * np.log(Z) + (1 - Y_vector) * np.log(1 - Z))
    
    return - sum / (2 * Y_vector.size)


def GradientDescentFunc(theta_vector, alpha, X_matrix, Y_vector, lamda):
    Z = SegmoidFunc(theta_vector, X_matrix)
    sum = np.sum(X_matrix.transpose() * (Z - Y_vector), axis=1)
    reguralization_term = (lamda / (2 * Y_vector.size)) * (np.sum(theta_vector@theta_vector, axis=0))
    
    return (alpha / Y_vector.size) * sum + reguralization_term


def DescalingFeatures(theta_vector):       # To reflect the effect of feature scalling by Mean Normalization method from the results
    
    return (theta_vector / (theta_vector.max() - theta_vector.min()))
        

def Train(file_name, iterations, alpha, lamda):
    
    Data_Frame = pd.read_csv(file_name)
    
    TrainingSet = np.array(Data_Frame)
    
    # features = X
    features_matrix = TrainingSet[:,0:-1]
    # values = Y
    values = TrainingSet[:,-1]
    
    
    theta_vector = np.ones(features_matrix[0].size)
    
    
    last_error = 0
    # iterating to change theta
    for i in range(iterations):
        try:
            if i < iterations - 1:
                if CostFunc(theta_vector, features_matrix, values) == np.inf:
                    print('Final Iteration #{}: Final Error = {} , Final Thetas = {}'.format(i + 1, last_error, theta_vector))
                    break
                else:
                    last_error = CostFunc(theta_vector, features_matrix, values)
                    print('Iteration #{}: Error = {} , Thetas = {}'.format(i + 1, last_error, theta_vector))
                    theta_vector -= GradientDescentFunc(theta_vector, alpha, features_matrix, values, lamda)
            else:
                print('Final Iteration #{}: Final Error = {} , Final Thetas = {}'.format(i + 1, last_error, theta_vector))
                
        except:
            print("Error: Logistic regression process, Error Explain: I can't learn!")
            break
        
    try:
        # theta_vector = DescalingFeatures(theta_vector)
        pd.DataFrame(theta_vector).to_csv("./results/results.csv", index=False, header=False)
    except:
        print("Error: DescalingFeatures process, Error Explain: Dividing on zero!")
    
    print(' ' + '-' * 29)
    print('| Training done successfully! |')
    print(' ' + '-' * 29)