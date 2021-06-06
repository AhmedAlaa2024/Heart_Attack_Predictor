import numpy as np
import pandas as pd
from numpy.core.fromnumeric import transpose
from pandas.core.frame import DataFrame
import Training_Script as train


def SegmoidFunc(theta_vector, X_matrix):        # The implementation of the segmoid function
    print('\nDebug =', theta_vector@X_matrix.transpose())
    return 1 / (1 + np.e**(-1 * theta_vector@X_matrix.transpose()))


def Predict(data):
    np.seterr(all='ignore')
    data.insert(0, 1)
    data = np.array(data).reshape(1, 14)
    theta_vector = np.array(pd.read_csv('./results/results.csv',header=None)).reshape(1,14)
    z = SegmoidFunc(theta_vector, data)

    print('\nZ =', z)
    if z == 1:
        print(' --------------------------------------------------------------------------------')
        print('| Sorry for telling that, but the patient has a probability to have heart attack |')
        print('| Result: Yes                                                                    |')
        print(' --------------------------------------------------------------------------------')
    else:
        print(' ----------------------------------------------------------------------')
        print("| Congratulations, You don't have the probability to have heart attach |")
        print('| Result: No                                                           |')
        print(' ----------------------------------------------------------------------')
        
Predict([57,1,2,128,229,0,0,150,0,0.4,1,1,3]) # Exact value = 0
Predict([63,1,3,145,233,1,0,150,0,2.3,0,0,1]) # Exact Value = 1