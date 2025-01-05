import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array



MODELS_DIR = '../models'
MODEL_NAME='final_classification_model.keras'
#file with classes
CLASSES_JSON = 'class_indices.json'
TARGET_SIZE = (320, 320)

def prepare_input(image_path):
    img=load_img(image_path, target_size=TARGET_SIZE)
    print(f"Loaded image in path: '{image_path}'")
    x = img_to_array(img)
    X = np.array([x])
    return preprocess_input(X)


# In[387]:


def decode_predictions(preds, class_indices):
    predictions=preds[0]
    #order in descendent order
    sorted_indices = np.argsort(predictions)[::-1]  

    top_n = 5
    top_classes = []
    top_scores = []

    index_to_class = {index: label for label, index in class_indices.items()}

    for i in range(top_n):
        index = sorted_indices[i]
        class_label = index_to_class[index] 
        score = predictions[index]
        top_classes.append((class_label, score))

    return top_classes
    
#Load model
model_file_path=f'{MODELS_DIR}/{MODEL_NAME}'
model = keras.models.load_model(model_file_path)
print("Model loaded...")


#load classes from json file
classes_json_path=f'{MODELS_DIR}/{CLASSES_JSON}'
class_indices={}
with open(classes_json_path, 'r') as json_file:
    class_indices = json.load(json_file)
print("Class indices loaded...")

# Prompt for image path
image_path = input("Enter the path to the image: ")

#running prediction
try:
    preds = model.predict(prepare_input(image_path))
    top_predictions = decode_predictions(preds, class_indices)
    print("Top 5 Predictions:")
    for class_label, score in top_predictions:
        print(f"{class_label}: {score:.4f}")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
