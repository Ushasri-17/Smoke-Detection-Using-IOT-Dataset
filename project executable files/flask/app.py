from flask import Flask, render_template, request, url_for
import numpy as np
import pickle


with open('knn.pkl','rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    #Get numerical inputs from the form
     temperature = float( request.form['Temperature[C]'])
     humidity = float( request.form['Humidity[%]'])
     tvoc = float( request.form['TVOC[ppb]'])
     eco2 = float( request.form['eCO2[ppm]'])
     raw_h2 = float( request.form['Raw H2'])
     pressure = float( request.form['Pressure[hPa]'])
     pm2_5 = float( request.form['PM2.5'])
     nc0_5 = float( request.form['NC0.5'])
     nc2_5 = float( request.form['NC2.5'])
     cnt = float( request.form['CNT'])

    #Create the final features array
     final_features = np.array([[temperature,humidity,tvoc,eco2,raw_h2,pressure,pm2_5,nc0_5,nc2_5,cnt]])
    

    #make the prediction

     prediction = model.predict(final_features)[0]
     print(prediction)


    #Set the prediction test based on model prediction

     if prediction == 0:
         return render_template('false.html')
    
    #Render the result template with the prediction text
     return render_template('submit.html')
     
if __name__ == '__main__':
        app.run(debug=True)