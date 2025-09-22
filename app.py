from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("ann_model0.h5")

# Danh sách class (em thay đúng theo dataset của mình)
class_names = ["hoa cuc", "hoa hong", "hoa lan", "hoa mat troi", "hoa sen" ]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"})

    # Xử lý ảnh
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((64, 64))  # resize theo input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    return jsonify({
        "class": class_names[pred_class],
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
