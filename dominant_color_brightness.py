import sys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colorsys

def rgb_to_hsl(r, g, b):
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

def hsl_to_rgb(h, s, l):
    return tuple(int(x * 255.0) for x in colorsys.hls_to_rgb(h, s, l))

def increase_brightness(color, delta):
    h, s, l = rgb_to_hsl(*color)
    l = min(1, l + delta)
    return hsl_to_rgb(h, s, l)

def get_dominant_color(image_path, n_colors=1, brightness_delta=0.1):
    image = Image.open(image_path)
    image = image.resize((150, 150)) # Resize image for faster processing
    image_array = np.array(image)
    image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))

    # Use KMeans clustering to find the most dominant color
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(image_array)

    # Get the RGB values of the dominant color
    dominant_color = kmeans.cluster_centers_[0]
    dominant_color = tuple(dominant_color.astype(int))

    # Increase the brightness of the dominant color
    brighter_color = increase_brightness(dominant_color, brightness_delta)

    return brighter_color

if __name__ == "__main__":
    image_path = input("請輸入圖片路徑：")

    try:
        brighter_color = get_dominant_color(image_path)
        print(f"圖片的主要色系（亮度增加）為：{brighter_color}")
    except Exception as e:
        print(f"讀取圖片時出錯: {e}")
