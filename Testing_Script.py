from datetime import time
from timeit import Timer
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import transpose
from pandas.core.frame import DataFrame
import Training_Script as train


def SegmoidFunc(theta_vector, X_matrix):        # The implementation of the segmoid function
    
    return 1 / (1 + np.e**(-1 * (theta_vector@X_matrix.transpose())))


def Test(file_name):
    
    Data_Frame = pd.read_csv(file_name)
          
    TestingSet = np.array(Data_Frame)

    # features = X
    features_matrix = TestingSet[:,0:-1]
    # values = Y
    values = TestingSet[:,-1]

    theta_vector = (np.array(pd.read_csv('./results/results.csv',header=None)).reshape(1,14))[0,:]
    results = list()
    true = 0
    false = 0
    for i in range(values.size):
        z = SegmoidFunc(theta_vector, features_matrix[i,:])
        if z == values[i]:
            results.append(1)
            true += 1
            print('Test #{}: Succeeded'.format(i))
        else:
            results.append(0)
            false += 1
            print('Test #{}: Failed'.format(i))
    
    print(' ' + '-' * 65)
    print('| The succeeded tests number: {}, and the failed tests number: {} |'.format(true, false))
    print("| Accuracy = {}                                    |".format((true / len(results)) * 100))
    print(' ' + '-' * 65)