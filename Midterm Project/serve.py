import pickle
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuration ---
model_file = 'et_model.bin'

# --- Initialization ---
app = Flask('Concrete Strength')


# load model 
with open(model_file, 'rb') as f_in:
            model_file = pickle.load(f_in)
print(f"Serving model '{model_file}' loaded successfully.")


# Post request
@app.route('/predict', methods=['POST'])

def predict():
    strength = request.get_json()

    X = pd.DataFrame([strength])
    y_pred = model_file.predict(X)

    result = { 
            'concrete mix strength': f"{round(float(y_pred), 2)} MPa"
    }
   
    return jsonify(result)


# --- Server Start ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)