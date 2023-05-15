from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import cv2
from io import BytesIO


model = None
interpreter = None
input_index = None
output_index = None


class_names = ["Nitrogen rich", "Phosphate rich"]

BUCKET_NAME = "soil-classifier"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

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

def get_class_response(ret):
    if ret == "Nitrogen rich":
        return [
            "- Nitrogen: High (70%)",
            "- Phosphate: Low (30%)",
            "- Potassium: Medium (50%)",
            "- Calcium: Low (20%)",
            "- Magnesium: Medium (40%)"
        ]
    elif ret == "Phosphate rich":
        return [
            "- Nitrogen: Low (30%)",
            "- Phosphate: High (70%)",
            "- Potassium: Medium (50%)",
            "- Calcium: Medium (50%)",
            "- Magnesium: Low (20%)"
        ]
    else: 
        return "There is an error"


def ping():
    return "Hello, I am alive"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/plant_classifier.h5",
            "/tmp/plant_classifier.h5",
        )
        model = tf.keras.models.load_model("/tmp/plant_classifier.h5")

    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_resized = Image.fromarray(image).resize((64, 64))
    preprocessed_img = preprocess_image(img_resized)
    input_img = np.expand_dims(preprocessed_img, axis=0)
    class_probs = model.predict(input_img)[0]
    class_index = np.argmax(class_probs)
    class_label = class_names[class_index]
    confidence = float(class_probs[class_index])

    response = {
        "class": class_label,
        "confidence": confidence,
        "details": get_class_response(class_label)
    }

    return jsonify(response)

