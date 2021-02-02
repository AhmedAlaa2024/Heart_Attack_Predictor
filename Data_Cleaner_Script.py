import pandas as pd
import numpy as np


def ScalingFeature(features):       # To apply feature scalling by Mean Normalization method
    
    maxmin = features.max(axis=0) - features.min(axis=0)
    mean = features.mean(axis=0)
    return (features - mean) / (maxmin)


def DataCleaning(file_name):
    file_path = './Data/' + file_name
    Data_Frame = pd.read_csv(file_path)
    DataSet = np.array(Data_Frame)

    TrainingSet = DataSet[0:int(DataSet.shape[0] * 0.7), :]
    TestingSet = DataSet[int(DataSet.shape[0] * 0.7):, :]


    TrainingSet_features = TrainingSet[:,0:-1]
    TrainingSet_values = TrainingSet[:,-1]
    TrainingSet_features = ScalingFeature(TrainingSet_features)
    TrainingSet = np.zeros(TrainingSet.shape)
    TrainingSet[:,0:-1] += TrainingSet_features
    TrainingSet[:,-1] += TrainingSet_values
    TrainingSet = np.insert(TrainingSet, 0, 1, axis=1)
    pd.DataFrame(TrainingSet).to_csv("./Data/TrainingSet.csv", index=False)
    print("Writting TrainingSet.csv ...")
    print("TrainingSet.csv: Done!")

    TestingSet_features = TestingSet[:,0:-1]
    TestingSet_values = TestingSet[:,-1]
    TestingSet_features = ScalingFeature(TestingSet_features)
    TestingSet = np.zeros(TestingSet.shape)
    TestingSet[:,0:-1] += TestingSet_features
    TestingSet[:,-1] += TestingSet_values
    TestingSet = np.insert(TestingSet, 0, 1, axis=1)
    pd.DataFrame(TestingSet).to_csv("./Data/TestingSet.csv", index=False)
    print("Writting TestingSet.csv ...")
    print("TestingSet.csv: Done!")
    

def Clean(file_name):
    print("Start Data Cleaning...")
    DataCleaning(file_name)
    print("Data Cleaning has been done successfully!") 