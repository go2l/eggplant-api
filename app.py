import os
import numpy as np
import cv2
from rembg import remove
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# פונקציה לחישוב כהות
def calculate_darkness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]  
    avg_l = np.mean(l_channel)
    darkness_score = 100 - int((avg_l / 255) * 100)
    return max(1, min(100, darkness_score))

# פונקציה לעיבוד תמונה
def process_image(image):
    removed_bg = remove(image)
    image_np = np.array(removed_bg)

    darkness = calculate_darkness(image_np)

    return {
        "רמת כהות": darkness
    }

# מסלול API
@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "לא נמצאה תמונה"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    result = process_image(image)

    return jsonify(result)

# **הוספת PORT תקין כדי למנוע בעיה ב-Render**
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # הגדרת פורט דינמי
    app.run(host="0.0.0.0", port=port, debug=True)
