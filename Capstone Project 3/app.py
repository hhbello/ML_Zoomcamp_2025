import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and metadata
model_file = 'xgb_tuned.pkl'
with open(model_file, 'rb') as f:
    model, feature_names, target_cols = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert incoming JSON to DataFrame with correct feature order
    try:
        df_input = pd.DataFrame([data])
        # Direct check for missing features if necessary, but XGBoost handles many cases
        # For simplicity, we assume the frontend sends correct keys
        
        # Ensure correct column order
        X = df_input[feature_names]
        
        # Predict
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)[0]
        
        result = {
            'prediction': target_cols[int(y_pred)],
            'probabilities': {target_cols[i]: float(y_prob[i]) for i in range(len(target_cols))}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
