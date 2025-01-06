import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


MODELS_DIR = '../models'
MODEL_NAME='final_classification_model.keras'
TFLITE_MODEL_NAME='classification_model.tflite'
CLASSES_JSON = 'class_indices.json'
TARGET_SIZE = (320, 320)

def prepare_input(image_path):
    with Image.open(image_path, 'r') as img:
        img = img.resize((TARGET_SIZE), Image.NEAREST)
    print(f"Loaded image in path: '{image_path}'")
    x = np.array(img, dtype='float32')
    X = np.array([x])
    return preprocess_input(X)    
    
def preprocess_input(x):
        x /= 127.5
        x -= 1.0
        return x    
    
def predict(interpreter, X):
    input_index = interpreter.get_input_details()[0]['index']
    output_index =interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)    


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
    
# Set up argument parsing
parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')
parser.add_argument('image_path', nargs='?', type=str, help='Path to the image file for prediction')    

# Parse arguments
args = parser.parse_args()

# Prompt for image path if not provided
if args.image_path is None:
    image_path = input("Enter the path to the image: ")
else:
    image_path = args.image_path
    
#create interpreter based on model
tflite_model_path=f'{MODELS_DIR}/{TFLITE_MODEL_NAME}'
interpreter = Interpreter(tflite_model_path)
interpreter.allocate_tensors()
print("Interpreter ready...")


#load classes from json file
classes_json_path=f'{MODELS_DIR}/{CLASSES_JSON}'
class_indices={}
with open(classes_json_path, 'r') as json_file:
    class_indices = json.load(json_file)
print("Class indices loaded...")

#running prediction
try:
    preds = predict(interpreter, prepare_input(image_path))
    top_predictions = decode_predictions(preds, class_indices)
    print("Top 5 Predictions:")
    for class_label, score in top_predictions:
        print(f"{class_label}: {score:.4f}")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
