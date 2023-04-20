import sys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image_path, n_colors=1):
    image = Image.open(image_path)
    image = image.resize((150, 150)) # Resize image for faster processing
    image_array = np.array(image)
    image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))
    
    # Use KMeans clustering to find the most dominant color
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(image_array)
    
    # Get the RGB values of the dominant color
    dominant_color = kmeans.cluster_centers_[0]
    
    return tuple(dominant_color.astype(int))

if __name__ == "__main__":
    image_path = input("請輸入圖片路徑：")
    
    try:
        dominant_color = get_dominant_color(image_path)
        print(f"圖片的主要色系為：{dominant_color}")
    except Exception as e:
        print(f"讀取圖片時出錯: {e}")

