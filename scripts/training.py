import os
import glob
import json
import warnings

import random
import pandas as pd
import numpy as np
import tensorflow as tf
import kagglehub

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.mixed_precision import set_global_policy


## CONSTANTS

# defining a target size for images
TARGET_SIZE = (320, 320)
#defining a batch size
BATCH_SIZE=16
#defining a max queue size (count of batches preloaded in the queue while the model is training)
MAX_QUEUE_SIZE=10
#path to models
MODELS_DIR = '../models'
MODEL_NAME='final_classification_model.keras'
#file with classes
CLASSES_JSON = 'class_indices.json'
#setting constants to fix hyperparameters
LEARNING_RATE=0.0001
INNER_LAYER_UNITS= 1280
DROPRATE=0.5





#Checking that Tensorflow detects a GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#Enabling Just-In-Time computation (during the execution of this notebook this seemed to speed up computations)
tf.config.optimizer.set_jit(True) 
#set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')
#setting seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
#hide some warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Checking for invalid filenames and getting information about images
def create_dataset(csv_file_path, images_dir):   
    #read dataset and skip some bad lines
    columns=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'productDisplayName']
    data = pd.read_csv(csv_file_path,  usecols=columns, on_bad_lines='skip')
    
    #add a new column to dataset, with the file name
    data['filename'] = data['id'].astype(str)+'.jpg'
    data['filename'] = data['filename'].str.strip() 
    data['filename'] = data['filename'].str.lower()  
    
    #delete from dataset those rows that don't have a corresponding image
    invalid_filenames = []
    for filename in data['filename']:
        file_path = os.path.join(images_dir, filename)
        if not os.path.exists(file_path):
            invalid_filenames.append(filename)
    data_cleaned= data[~data['filename'].isin(invalid_filenames)]
    
    # Filter the article types to keep only those with more than 4 samples 
    #(we need at least 1 sample in each dataframe: train, val and test and with a small quantity of samples, it is difficult to achieve)
    sub_category_counts =data.groupby('articleType').size().reset_index(name='count')
    valid_article_types = sub_category_counts[sub_category_counts['count'] >= 4]['articleType']
    return data_cleaned[data_cleaned['articleType'].isin(valid_article_types)]
    
def make_model(count_classes, learning_rate, inner_layer_units, droprate):
    
    #clear any unused session
    tf.keras.backend.clear_session()

    #creating a base model
    input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    base_model =MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)
    # create a model from the base model
    base_model.trainable=False
    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    
    #pooling layer
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    #dense layer 1
    inner = keras.layers.Dense(inner_layer_units, activation='relu')(vectors)

    #dropout for inner layer 1
    drop = keras.layers.Dropout(droprate)(inner)
    
    #dense layer 2
    outputs = keras.layers.Dense(units=count_classes, activation='softmax')(drop)

    #final model
    model = keras.Model(inputs, outputs)

    #compile de model
    optimizer =keras.optimizers.Adam(learning_rate=learning_rate)
    loss=keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model



#download the dataset from kagglehub
path_data_files = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
print("Path to dataset files:", path_data_files)

dataset_dir = f'{path_data_files}/fashion-dataset'
csv_file_path = F'{dataset_dir}/styles.csv'
images_dir = f'{dataset_dir}/images'
    
#create the dataframe
data = create_dataset(csv_file_path, images_dir)

#get number of classes in the dataset
count_classes = data['articleType'].nunique()

#split the data into training, validation and test datasets
full_train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['articleType'])

print(f'Total images that will be used as input: {len(data)}')
print(f'Total images for training: {len(full_train_df)}')
print(f'Total images for testing: {len(test_df)}\n')


# # Training the model 
print ('Training the model. Hold on, this make take a while to complete...\n')
# calculating how many workers threads for loading data
num_cores = os.cpu_count()
workers = int(num_cores / 2)
use_multiprocessing=False
if workers>1:
   use_multiprocessing=True

train_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = train_image_gen.flow_from_dataframe(
    dataframe=full_train_df,
    directory=images_dir, 
    x_col='filename',  
    y_col='articleType', 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,                
    workers = workers,
    use_multiprocessing=use_multiprocessing,
    max_queue_size=MAX_QUEUE_SIZE
)


model = make_model(count_classes=count_classes, learning_rate=LEARNING_RATE, 
                    inner_layer_units=INNER_LAYER_UNITS, droprate=DROPRATE)
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=None
)

# # Evaluating the model 
test_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_image_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=images_dir, 
    x_col='filename',  
    y_col='articleType', 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    shuffle=False,
    workers = workers,
    use_multiprocessing=use_multiprocessing,
    max_queue_size=MAX_QUEUE_SIZE    
)
test_loss, test_accuracy = model.evaluate(test_gen)
print (f"The accuracy of the model is {test_accuracy}")


# # Saving the model 
model_file_path=f'{MODELS_DIR}/{MODEL_NAME}'
model.save(f'{model_file_path}')
print(f"Model saved as {model_file_path}")

## Saving classes
class_indices = train_gen.class_indices
classes_json_path=f'{MODELS_DIR}/{CLASSES_JSON}'
with open(classes_json_path, 'w') as json_file:
    json.dump(class_indices, json_file)

print(f'Class indices saved to: {classes_json_path}')
