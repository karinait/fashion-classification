
# Fashion Classification Project

This project involves building a convolusional neural network to predict the type of a clothing item from a picture. The prediction service could be used from any retail application where the user can update a picture of an item and the system proposes an item category from the predefined ones.

## Table of Contents

- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Run the Prediction Service with Docker](#run-the-prediction-service-with-docker)
- [Predictions with a Python script](#prediction-with-a-python-script)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [AWS Lambda](#aws-lambda)

---


## Dataset

The were two datasets used for this projects, both sourced from Kaggle. 

- **<a name="largedata">[Large Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)</a>**

- **<a name="smalldata">[Small Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)</a>**

---

## Technologies Used

- Python (with libraries such as pandas, numpy, matplotlib, etc.)
- Tensorflow
- Tensorflow runtime
- Jupyter Notebook 
- Docker
- AWS Lambda

---

## Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites
- Install [Python 3.9](https://wiki.python.org/moin/BeginnersGuide/Download)  
- Install [Docker](https://www.docker.com/get-started)
- Install [Pipenv] (https://pypi.org/project/pipenv/#installation) (optional)

---

## Run the Prediction Service with Docker

1. **Clone the Repository**
   ```bash
   git clone https://github.com/karinait/fashion-classification.git
   cd fashion-classification
   ```
2. **Build the Docker Image**
   Use the provided `Dockerfile` to build the Docker image:
   ```bash
   cd fashion-classifier/app
   docker build -t fashion-classifier .
   ```

3. **Run the Docker Container**
   Start the container to run the project:
   ```bash
   docker run -it --rm -p8080:8080 fashion-classifier
   ```

4. **Access the Service Locally**
   Use curl to post an image url to the service running at localhost:
   ```
   curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" \
	-d '{ "url":  "http://bit.ly/mlbookcamp-pants"}'
   ```
   The response should be something like:
   
   ```
	{'Casual Shoes': 0.2870946526527405, 'Flats': 0.06451930850744247, 'Formal Shoes': 0.4267609715461731, 'Heels': 0.1557253748178482, 'Sandals': 0.05419463291764259}   
   ```   
 
   The category with the highest score would correspond to the most probable item type corresponding to the image evaluated. In this example the category would be "Formal Shoes".   

## Predictions with a Python script

   In case you want to run predictions against some images stored locally in the repository, you could use the predict.py script in order to do that. To do so, please follow these steps:
   
   
1. **Prepare the Environment**

   Create a virtual environment with pipenv for python 3.9 and install all the dependencies  
   ```bash
   cd fashion-classification
   pipenv --python 3.9
   pipenv install
   ```  

2. **Use the predict.py script to run predictions**

   Activate the pipenv environment and from there use the predict.py script in fashion-classification/scripts
   ```bash
   pipenv shell
   cd scripts
   python predict.py
   ```   
   
   The script will request a path to the image and you can type in the path to any of the images inside the fashion-classification/dataset/test-images folder. For example: ../dataset/test-images/2639.jpg (note: enter the path without any "")
   
   The response should be something like:
   
   ```bash
	Top 5 Predictions:

	{'Casual Shoes': 0.2870946526527405, 'Flats': 0.06451930850744247, 'Formal Shoes': 0.4267609715461731, 'Heels': 0.1557253748178482, 'Sandals': 0.05419463291764259}
   ```
   
---

## Project Structure

The relevant folders and files in this project are the ones listed below:

```
fashion-classification/
│
├── dataset/		# Contains the small version of the dataset and some image for testing
│   ├── fashion-product-images-small/		# Folder with the small images dataset downloaded from kaggle
│		├── images		# Images as .jpg
│		├── styles.csv		# Csv with all the information about the images
│   ├── test-images/		# Folder with large images for test	
├── fashion-classifier/		# Folder created with AWS SAM CLI for the deployment to AWS Lambda
│   ├── app/		# Folder with all the necessary files for deployment
│		├── Dockerfile		# Docker configuration
│		├── app.py		# Script with lambda function
│		├── classification_model.tflite		# Trained Tflite model
│		├── class_indices.json		# Json with the name and indices of the classes used to train the model
│		├── requirements.txt		# List of dependencies that must be installed in the Docker container		
├── models/		# Models created during evaluation and training of different CNN
├── notebooks/		# Jupyter notebooks for analysis
├── Pipfile		# Project top-level requirements (to be used with pipenv)
├── Pipfile.lock		# Required dependencies with specific versions (to be used with pipenv)
├── README.md		# Project documentation
├── video/		# Folder that shows the lambda function being tested in AWS Lambda
└── .gitignore		# Git ignore file
```

---

## Jupyter notebooks

The jupyter notebooks for this project were run locally on a machine with a GPU where tensorflow was installed. The explanation on how to setup the environment for tensorflow is out of the scope of this document

The jupyter notebooks in this project were used for the Exploratory Data Analysis, Training and Inference. Here is a description of every notebook:

-**notebook_1_eda_small_images.ipynb**

This notebook was used for the Exploratory Data Analysis of the [Small Images Dataset](#smalldata) and to train several neural networks and tune their parameters.

The final result in this notebook is the selection of the final model to train on the larger dataset


-**notebook2_training_large_dataset.ipynb**

This notebook was used mainly to train the model selected in the notebook1 on the [Large Images Dataset](#largedata) 

-**predict.ipynb**

This notebook was used to make predictions by using the model trained in the notebook 2 and using tensorflow for inference

-**predict_tflite.ipynb**

This notebook was used to create a tflite version of the model and use that version to make predictions through tflite runtime

---

## AWS Lambda

For the deployment of the project to AWS Lambda, I used the AWS SAM CLI to create a baseline and then adjusted the files as needed. The function was deployed sucessfully and there is a video in the /video folder showing a simple test of this function.

