import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#pickle to reade model.py pickle file created

#it ill create my flask app
app = Flask(__name__)
#we need to load the pickle
model = pickle.load(open('model.pkl', 'rb'))

#route function of the Flask Class is decorator
@app.route('/')
def home():
    return render_template('index.html')

#predict post method providing some features to model.pkl file so that model will take some input and give some output
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features) 
    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

#main function which will run whole flask
if __name__ =="__main__":
    app.run(debug=True)