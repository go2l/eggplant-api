import os
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, jsonify
from rembg import remove
from PIL import Image

app = Flask(__name__)

# פונקציה לחישוב כהות
def calculate_darkness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]  # ערוץ הבהירות (L)
    avg_l = np.mean(l_channel)
    darkness_score = 100 - int((avg_l / 255) * 100)
    return max(1, min(100, darkness_score))

# פונקציה לחישוב אחוזי צבעים
def calculate_color_percentages(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masked_hsv = hsv[mask > 0]

    if masked_hsv.size == 0:
        return {"שחור": 0, "סגול כהה": 0, "סגול בהיר": 0, "חום": 0}

    total_pixels = masked_hsv.shape[0]
    color_percentages = {}

    color_ranges = {
        "שחור": ((0, 0, 0), (180, 255, 50)),
        "סגול כהה": ((110, 50, 20), (160, 255, 100)),
        "סגול בהיר": ((110, 40, 100), (170, 255, 255)),
        "חום": ((5, 40, 20), (40, 255, 200)),
    }

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        color_mask = (
            (masked_hsv[:, 0] >= lower[0]) & (masked_hsv[:, 0] <= upper[0]) &
            (masked_hsv[:, 1] >= lower[1]) & (masked_hsv[:, 1] <= upper[1]) &
            (masked_hsv[:, 2] >= lower[2]) & (masked_hsv[:, 2] <= upper[2])
        )

        color_pixels = np.count_nonzero(color_mask)
        color_percentages[color] = round((color_pixels / total_pixels) * 100, 2)

    return color_percentages

# עיבוד תמונה
def process_image(image):
    removed_bg = remove(image)
    image_np = np.array(removed_bg)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
    avg_color = np.mean(masked_image[mask > 0], axis=0).astype(int)

    darkness = calculate_darkness(masked_image)
    color_percentages = calculate_color_percentages(image_np, mask)

    return darkness, avg_color.tolist(), color_percentages

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "לא נמצאה תמונה"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    darkness, avg_color, color_percentages = process_image(image)

    response = {
        "רמת כהות": darkness,
        "צבע ממוצע": f"RGB({avg_color[0]}, {avg_color[1]}, {avg_color[2]})",
        "אחוזי צבעים": color_percentages
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # יציאה דינמית לפי Render
    app.run(host="0.0.0.0", port=port, debug=True)
