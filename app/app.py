from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        # Prediction
        probs = model.predict_proba(features)[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Optional: confidence check
        if confidence < 0.6:
            return jsonify({
                "message": "Model not confident",
                "confidence": confidence
            })

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # IMPORTANT for deployment (Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)