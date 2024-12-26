import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random

# Load the pre-trained model
model_name = "trash_sorting_model_v2.keras"
model = load_model(model_name)

data_dir = './dataset-resized'
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return "Unknown", 0.0
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction)  # Confidence is the highest probability
    return predicted_class, confidence

# Function to load random images from all categories
def load_random_images(data_dir, categories, num_images=5):
    images = []
    image_paths = []
    
    # Collect all images from all categories
    all_image_paths = []
    for category in categories:
        path = os.path.join(data_dir, category)
        img_files = os.listdir(path)
        for img_file in img_files:
            img_path = os.path.join(path, img_file)
            all_image_paths.append(img_path)

    # Randomly select 'num_images' images from all categories
    selected_paths = random.sample(all_image_paths, num_images)
    
    # Read and resize selected images
    for img_path in selected_paths:
        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_array is not None:  # Check if the image was loaded properly
            resized_img = cv2.resize(img_array, (128, 128))
            images.append(resized_img)
            image_paths.append(img_path)  # Store the actual path for prediction

    return np.array(images), image_paths

# Load a batch of random images and their labels
num_images = 5  # Number of images to display
random_images, random_image_paths = load_random_images(data_dir, categories, num_images)

# Normalize images
random_images = random_images / 255.0

# Make predictions with confidence
predictions = [predict_image(image_path) for image_path in random_image_paths]

# Plot images and their predictions with confidence
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
for i in range(num_images):
    ax = axes[i]
    ax.imshow(random_images[i])
    ax.axis('off')
    predicted_class, confidence = predictions[i]
    ax.set_title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%")
plt.show()
