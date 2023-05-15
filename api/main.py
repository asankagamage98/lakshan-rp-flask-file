from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2


app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model("../saved_models/plant_classifier.h5")

CLASS_LABELS = ["Nitrogen rich", "Phosphate rich"]

def preprocess_image(image):
    image = image.convert("RGB")
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, (64, 64))
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image_resized, image_resized, mask=cv2.bitwise_not(mask))
    return result

@app.route("/ping")
def ping():
    return "Hello, I am alive"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

def getSoilClass(ret):
    if ret == "Nitrogen rich":
        return ("- Nitrogen: High (70%)",
                "- Phosphate: Low (30%)",
                "- Potassium: Medium (50%)",
                "- Calcium: Low (20%)",
                "- Magnesium: Medium (40%)")
    elif ret == "Phosphate rich":
        return ("- Nitrogen: Low (30%)",
                "- Phosphate: High (70%)",
                "- Potassium: Medium (50%)",
                "- Calcium: Medium (50%)",
                "- Magnesium: Low (20%)")

    else: 
        return "There is an error"
    

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_resized = Image.fromarray(image).resize((64, 64))
    preprocessed_img = preprocess_image(img_resized)
    input_img = np.expand_dims(preprocessed_img, axis=0)
    class_probs = MODEL.predict(input_img)[0]
    class_index = np.argmax(class_probs)
    class_label = CLASS_LABELS[class_index]
    confidence = float(class_probs[class_index])
    
    response = {
        "class": class_label,
        "confidence": confidence,
        "details": getSoilClass(class_label)
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="localhost", port=9000)
