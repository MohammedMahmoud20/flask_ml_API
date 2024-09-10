from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the Gradient Boosting model
GBmodel = joblib.load('model/logistic_regression_model.pkl')

# Assuming you used MinMaxScaler for scaling the features
scaler = MinMaxScaler()

# Define a route for predicting autism traits using the Gradient Boosting model
@app.route('/ml', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Convert the input data into a DataFrame for processing
    # Assuming the input JSON contains keys like 'A1', 'A2', etc., matching the feature names
    input_data = pd.DataFrame([data])

    # Specify the same order of features as during model training
    order = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age','Qchat-10-Score', 'Sex', 'Jaundice', 'Family_mem_with_ASD','Ethnicity_Latino', 'Ethnicity_Native Indian', 'Ethnicity_Others',
'Ethnicity_Pacifica', 'Ethnicity_White European', 'Ethnicity_asian',
'Ethnicity_black', 'Ethnicity_middle eastern', 'Ethnicity_mixed',
'Ethnicity_south asian', 'Who completed the test_Health care professional',
'Who completed the test_Others', 'Who completed the test_Self','Who completed the test_family member']

    # Ensure the data has the same columns as the training data (and same order)
    input_data = input_data[order]

    # Preprocess the data (scaling)
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the Gradient Boosting model
    prediction = GBmodel.predict(input_data_scaled)

    # Prepare the response
    response = {
        'prediction': int(prediction[0])  # Convert the prediction to a simple integer
    }

    # Return the prediction as a JSON response
    return jsonify(response)

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
