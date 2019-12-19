import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

#fill na values of experience column with 0
dataset['experience'].fillna(0, inplace=True)

#fill test_score column na entries with mean 
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

#create all independent columns(experience, test_score, interview_score) on X variable
X = dataset.iloc[:, :3]

#function to convert experience values in integers
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
                 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return word_dict[word]
#applying it to experience feature column
X['experience']= X['experience'].apply(lambda x: convert_to_int(x))

#dependent feature
y= dataset.iloc[:, -1]

#create linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with training data
regressor.fit(X, y)

#saving model to disk using pickle library in write bytes mode
#important - model.pkl file will be deployed in heroku enviornment 
pickle.dump(regressor, open('model.pkl', 'wb'))


#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
