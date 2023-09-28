#import streamlit as st

#st.header('Welcome to Streamlit!')

from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from heartfailure.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age =float(request.form['age'])
            anaemia =float(request.form['anaemia'])
            creatinine_phosphokinase =float(request.form['creatinine_phosphokinase'])
            diabetes =float(request.form['diabetes'])
            ejection_fraction =float(request.form['ejection_fraction'])
            high_blood_pressure =float(request.form['high_blood_pressure'])
            platelets =float(request.form['platelets'])
            serum_creatinine =float(request.form['serum_creatinine'])
            serum_sodium =float(request.form['serum_sodium'])
            sex =float(request.form['sex'])
            smoking =float(request.form['smoking'])
       
         
            data = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)