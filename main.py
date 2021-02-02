import os
import numpy as np
import Training_Script as train
import Testing_Script as test
import Data_Cleaner_Script as Cleaner
from timeit import default_timer as timer 

def Request_filepath():
    error = None
    while(True):
        if error == None:
            try:
                file_name = input('Enter the filename : ') # getting filename
                file_path = './Data/' + file_name
                Data_Frame = np.genfromtxt(file_path, delimiter = ',') # loading the file into a numpy 2D array
                break
            except:
                error = "File doesn't exist!"
                continue
        elif error == "File doesn't exist!":
            print(' ' + '-' * 91)
            print("| The file doesn't exist! Please, Make sure that the file exists and re-enter the file name |")
            print(' ' + '-' * 91)
            try:
                file_name = input('Enter the filename : ') # getting filename
                file_path = './Data/' + file_name
                Data_Frame = np.genfromtxt(file_path, delimiter = ',') # loading the file into a numpy 2D array
                error = None
                break
            except:
                error = "File doesn't exist!"
                continue
    return file_path


def Request_Train_Startup():
    iterations = int(input("num_of_iterations = "))
    alpha = float(input("alpha = "))
    lamda = float(input("lamda = "))
    
    return iterations, alpha, lamda

def CalculateExecutionTime(start):
    
    print("Execution time: {} seconds".format(timer() - start))
    

print("Hello, I am Ahmed, the heart attack predictor. Nice to meet you! How can I help you?")
while(True):
    
    print('[1] You want me to clean and splite a data file for you into TrainingSet.csv and TestingSet.csv?')
    print('[2] You want to train me?')
    print('[3] You want to test me?')
    print('[4] You want me to predict the heart attack probability for a patient?')
    print("[5] Thanks, I don't need something.")
    
    option = input('Please, Enter the number of option: ')
    cleaned = False
    file_path = ''
    
    if option == '1':
        file_path = Request_filepath()
        Cleaner.Clean(file_path)
        cleaned = True
        
        response = input('Do you want another service? (y/n): ')
        if response == 'y':
            continue
        elif response == 'n':
            print("Thank you sir. Please, visit me again!")
            break
        else:
            print("I don't understand so I will suppose you said 'y'")
            continue
    elif option == '2':
        if cleaned:
            response = input("Do you want to use the TrainingSet.csv that is generated from the cleaned file '{}'? (y/n): ".format(file_path))
            if response == 'y':
                file_path = './Data/TrainingSet.csv'
            elif response == 'n':
                file_path = Request_filepath()
            else:
                print("I don't understand so I will suppose you said 'y'")
                file_path = './Data/TrainingSet.csv'
        else:
            response = input('Do have a TrainingSet file? (y,n): ')
            if response == 'y':
                file_path = Request_filepath()
            if response == 'n':
                print('Then, I will use my own.')
                Cleaner.Clean('./Data/heart.csv')
                file_path = './Data/TrainingSet.csv'
            else:
                print("I don't understand so I will suppose you said 'n'")
                print('Then, I will use my own.')
                Cleaner.Clean('./Data/heart.csv')
                file_path = './Data/TrainingSet.csv'
            
        iterations, alpha, lamda = Request_Train_Startup()
        start = timer()
        train.Train(file_path, iterations, alpha, lamda)
        CalculateExecutionTime(start)
        response = input('Do you want another service? (y/n): ')
        if response == 'y':
            continue
        elif response == 'n':
            print("Thank you sir. Please, visit me again!")
            break
        else:
            print("I don't understand so I will suppose you said 'n'")
            print("Thank you sir. Please, visit me again!")
            break
    elif option == '3':
        if cleaned:
            response = input("Do you want to use the TestingSet.csv that is generated from the cleaned file '{}'? (y/n): ".format(file_path))
            if response == 'y':
                file_path = './Data/TestingSet.csv'
            elif response == 'n':
                file_path = Request_filepath()
            else:
                print("I don't understand so I will suppose you said 'y'")
                file_path = './Data/TestingSet.csv'
        else:
            response = input('Do have a TestingSet file? (y,n): ')
            if response == 'y':
                file_path = Request_filepath()
            if response == 'n':
                print('Then, I will use my own.')
                Cleaner.Clean('./Data/heart.csv')
                file_path = './Data/TestingSet.csv'
            else:
                print("I don't understand so I will suppose you said 'n'")
                print('Then, I will use my own.')
                Cleaner.Clean('./Data/heart.csv')
                file_path = './Data/TestingSet.csv'
                
        start = timer()
        test.Test(file_path)
        CalculateExecutionTime(start)
        response = input('Do you want another service? (y/n): ')
        if response == 'y':
            continue
        elif response == 'n':
            print("Thank you sir. Please, visit me again!")
            break
        else:
            print("I don't understand so I will suppose you said 'n'")
            print("Thank you sir. Please, visit me again!")
            break
    elif option == '4':
        print("\nSorry, My developer is still working on this service. Please, choose another sevice!")
        continue
        # headers = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        #         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        # data = list()
        # for header in headers:
        #     data.append(float(input("Enter the patient's {}: ".format(header))))
        # response = input('Do you want another service? (y/n): ')
        # if response == 'y':
        #     continue
        # elif response == 'n':
        #     print("Thank you sir. Please, visit me again!")
        #     break
    elif option == '5':
        response = input('Are you sure that you want to leave me? (y/n): ')
        if response == 'y':
            print("Thank you sir. Please, visit me again!")
            break
        elif response == 'n':
            print("\nI am happy! what do you want me to do?")
            continue
        else:
            print("I don't understand so I will suppose you said 'n'")
            print("\nI am happy! what do you want me to do?")
            continue

    else:
        print('\nSorry, There is no option like {}!'.format(option))
        continue 

os.remove('./Data/TrainingSet.csv')
os.remove('./Data/TestingSet.csv')