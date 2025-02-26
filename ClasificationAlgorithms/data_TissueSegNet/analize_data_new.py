import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import exposure

def analyze_dataset(image_path, mask_path, output_path="assets_analysis_new"):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images_analysis"), exist_ok=True)
    
    masks = [np.array(Image.open(os.path.join(mask_path, f)).convert("P"), dtype=np.uint8) 
             for f in os.listdir(mask_path) if f.endswith('.png')]
    images = [cv2.imread(os.path.join(image_path, f)) 
              for f in os.listdir(image_path) if f.endswith('.png')]
    
    # --- Análisis de distribución de clases en máscaras ---
    class_counts = {i: [] for i in range(4)}
    total_pixels = np.prod(masks[0].shape)
    
    for mask in masks:
        for i in range(4):
            class_counts[i].append(np.sum(mask == i) / total_pixels)
    
    avg_proportions = {i: np.mean(class_counts[i]) for i in range(4)}
    
    with open(os.path.join(output_path, "masks_analysis", "class_distribution.json"), "w") as f:
        json.dump(avg_proportions, f, indent=4)
    
    # Histograma de proporciones de clases
    bins = np.linspace(0, 1, 11)
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.hist(class_counts[i], bins=bins, alpha=0.7, label=f'Class {i}')
    plt.xlabel("Proportion bins")
    plt.ylabel("Image count")
    plt.title("Histogram of class proportions")
    plt.legend()
    plt.savefig(os.path.join(output_path, "masks_analysis", "hist_class_distribution.png"))
    plt.close()
    
    # --- Análisis de imágenes ---
    brightness_metrics = []
    texture_metrics = []
    
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        brightness_metrics.append(brightness)
        
        gray = rgb2gray(img)
        glcm = graycomatrix((gray * 255).astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        texture_metrics.append({"contrast": contrast, "dissimilarity": dissimilarity})
    
    result_metrics = {
        "average_brightness": np.mean(brightness_metrics),
        "average_texture": {
            "contrast": np.mean([t["contrast"] for t in texture_metrics]),
            "dissimilarity": np.mean([t["dissimilarity"] for t in texture_metrics])
        }
    }
    
    with open(os.path.join(output_path, "images_analysis", "image_metrics.json"), "w") as f:
        json.dump(result_metrics, f, indent=4)
    
    # Histograma de brillo
    plt.figure(figsize=(10, 6))
    plt.hist(brightness_metrics, bins=20, alpha=0.7)
    plt.xlabel("Brightness")
    plt.ylabel("Frequency")
    plt.title("Histogram of Image Brightness")
    plt.savefig(os.path.join(output_path, "images_analysis", "hist_brightness.png"))
    plt.close()
    print("Analysis completed.")

# Uso
def main():
    # analyze_dataset("data_padded/images", "data_padded/masks", output_path="assets_analysis_new")
    # analyze_dataset("data_padded/test_images", "data_padded/test_masks", output_path="assets_analysis_new/tvt/test")
    # analyze_dataset("data_padded/val_images", "data_padded/val_masks", output_path="assets_analysis_new/tvt/val")
    # analyze_dataset("data_padded/train_images", "data_padded/train_masks", output_path="assets_analysis_new/tvt/train")
    analyze_dataset("data_padded/trainval_images", "data_padded/trainval_masks", output_path="assets_analysis_new/tvt/trainval")

if __name__ == "__main__":
    main()