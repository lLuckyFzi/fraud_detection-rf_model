from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
import sys

app = Flask(__name__)

CORS(app)

try:
    model = joblib.load('fraud_rf_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
except Exception as e:
    print("Error loading model or scaler:", e)
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        TRESHOLD_FRAUD = 0.5

        data = request.get_json()
        expected_features = [
            'amount',
            'avg_amount_30d',
            'tx_count_24h',
            'failed_tx_count_24h',
            'distance_from_home',
            'channel_Web_Portal'
        ]

        df_input = pd.DataFrame([data], columns=expected_features)
        
        X_scaled = scaler.transform(df_input)
        X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)

        probabilities = model.predict_proba(X_scaled_df)[0]
        fraud_risk_score = probabilities[1]

        is_fraud = int(fraud_risk_score > TRESHOLD_FRAUD)

        return jsonify({
            "status": "success",
            "fraud_probability_percent": round(fraud_risk_score * 100, 2),
            "is_fraud": is_fraud,
            "action_recommended": "BLOCK" if is_fraud else "ALLOW"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Terjadi kesalahan pada pemrosesan ML.",
            "error_details": str(e),
            "trace": traceback.format_exc()
        }), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)