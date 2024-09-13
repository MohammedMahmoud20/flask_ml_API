from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model_path = 'model/my_ml_v2_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data sent as JSON
    if 'features' not in data:
        return jsonify({"error": "Missing features in request"}), 400
    # Assuming the input is a list of features
    features = np.array(data['features']).reshape(1, -1)
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    # Return the prediction as a JSON response
    return jsonify({"prediction": int(prediction[0])})



if __name__ == '__main__':
    app.run(debug=True)
