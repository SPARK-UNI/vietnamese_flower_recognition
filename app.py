from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

MODEL_PATH = r"your_model_path/flower_cnn.h5"
model = load_model(MODEL_PATH)

class_names = ['Hoa Ly', 'Hoa Sen', 'Hoa Lan', 'Hoa Hướng Dương', 'Hoa Tulip']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))  

    img_array = np.array(img) 
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = round(float(np.max(preds[0]) * 100), 2)

    return jsonify({
        "class": class_names[class_idx],
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
