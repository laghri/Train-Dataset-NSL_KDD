import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,make_response
import pickle
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__, static_folder='static')

model = load_model('NSS_KDD_Model.h5')
model2=load_model('NSS_KDD_Model2.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/test")
def about():
    return render_template("test.html")

@app.route('/predict',methods=['POST'])
def predict():
        file = request.files["fileToUpload"]
        if file:
           
            df = pd.read_csv(file)
         # Make predictions using the model and the DataFrame
            predictions = model2.predict(df)
            print(predictions)
        # Add the predictions to the DataFrame as a new column
            
            df['prediction'] =np.around(predictions) 
            tf.constant(df, dtype=tf.float32)
        # Save the DataFrame as a CSV file
            # result = df.to_excel("prediction_results.xlsx",index=False)
            excel_file = pd.ExcelWriter('output.xlsx')
            df.to_excel(excel_file, index=False)
            excel_file.save()

    # Return the Excel file for download
            response = make_response(open('output.xlsx', 'rb').read())
            response.headers.set('Content-Type', 'application/vnd.ms-excel')
            response.headers.set('Content-Disposition', 'attachment', filename='output.xlsx')
        return response

            
@app.route('/test_input')
def test():
    return render_template('test_input.html')
@app.route('/predict_input',methods=['POST'])
def predict_input():
    
    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print( final_features)
    work=pd.DataFrame(final_features )

    prediction = model2.predict(work)
    print(prediction)
    output = np.around(prediction[0], 2)
    print(bool(output[0]))
    v=True

    return render_template('test_input.html', condition=bool(output[0]),v=v,list=int_features)
if __name__ == "__main__":
    app.run(debug=True)