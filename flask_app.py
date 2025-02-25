from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Correct FastAPI backend URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            features = request.form.get("features")
            features = list(map(float, features.split(",")))  # Convert input to list of floats
            
            # Send request to FastAPI
            response = requests.post(FASTAPI_URL, json={"features": features})
            response_data = response.json()

            if response.status_code == 200:
                prediction = response_data.get("prediction", "Unknown")
            else:
                error_message = response_data.get("detail", "Error processing request")
        except Exception as e:
            error_message = f"Invalid input: {str(e)}"

    return render_template("index.html", prediction=prediction, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Running Flask on port 5001
