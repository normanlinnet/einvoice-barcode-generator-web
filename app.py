from flask import Flask, request, jsonify, send_from_directory
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import os

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/dominant-color', methods=['POST'])
def dominant_color():
    image_data = request.form.get('image')
    if not image_data:
        return jsonify({"error": "圖片資料未提供"}), 400

    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": f"讀取圖片時出錯: {e}"}), 400

    try:
        brighter_color = get_dominant_color(image)
        return jsonify({"dominant_color": brighter_color})
    except Exception as e:
        return jsonify({"error": f"處理圖片時出錯: {e}"}), 500

def get_dominant_color(image, n_colors=1, brightness_delta=0.1):
    image = image.resize((150, 150))
    image_array = np.array(image)
    image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(image_array)

    dominant_color = kmeans.cluster_centers_[0]
    dominant_color = tuple(dominant_color.astype(int))

    brighter_color = increase_brightness(dominant_color, brightness_delta)
    return brighter_color

def increase_brightness(color, delta):
    h, s, l = rgb_to_hsl(*color)
    l = min(1, l + delta)
    return hsl_to_rgb(h, s, l)

def rgb_to_hsl(r, g, b):
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

def hsl_to_rgb(h, s, l):
    return tuple(int(x * 255.0) for x in colorsys.hls_to_rgb(h, s, l))

if __name__ == '__main__':
    app.run(debug=True)
