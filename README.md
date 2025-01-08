
# Fashion Classification Project

This project aims to develop a convolutional neural network (CNN) to predict the type of clothing item from an image. The model is designed to classify images into predefined categories of fashion items, such as shirts, shoes, and accessories. The prediction service can be integrated into any retail application, allowing users to upload an image of a clothing item, after which the system proposes the most relevant category based on the visual content of the image. This functionality can enhance user experiences by automating product categorization in online retail environments.

---


## Table of Contents

- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Run the Prediction Service with Docker](#run-the-prediction-service-with-docker)
- [Other Ways to Run Predictions](#other-ways-to-run-predictions)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Scripts](#scripts)
- [AWS Lambda](#aws-lambda)

---


## Dataset

There were two datasets used for this project, both sourced from Kaggle. 

- **[Large Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)**

- **[Small Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)**

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

To get started with this project and run the prediction service locally, you'll need to set up your development environment and ensure a few prerequisites are installed. Follow the steps below to get a local copy of the repository, build the necessary Docker image, and run the service to begin making predictions.

### Prerequisites
- Install [Python 3.9](https://wiki.python.org/moin/BeginnersGuide/Download)  
- Install [Docker](https://www.docker.com/get-started)
- Install [Pipenv](https://pypi.org/project/pipenv/#installation) (optional)

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
   Start the container to run the prediction service:
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
 
   The category with the highest score corresponds to the most probable item type for the image. In this example, the most suitable item type for the image would be "Formal Shoes".   

   
---


## Other Ways to Run Predictions

If you want to run predictions on alternatives ways, you can use the predict.py directly or as a flask application. Both methods are explained below.


1. **Prepare the Environment**

Create a virtual environment with Pipenv for Python 3.9 and install all the dependencies: 

```bash
cd fashion-classification
pipenv --python 3.9
pipenv install
```  

2. **Run predictions**

The first step is to activate the Pipenv environment and navigate to the scripts folder:

```bash
pipenv shell
cd scripts
```   	
	
From there you can:

a)***Run the script.py script***

```bash
python predict.py "../dataset/test-images/2639.jpg"
```   	

The response will look like this:
	
```bash
Top 5 Predictions:

{'Casual Shoes': 0.2870946526527405, 'Flats': 0.06451930850744247, 'Formal Shoes': 0.4267609715461731, 'Heels': 0.1557253748178482, 'Sandals': 0.05419463291764259}   
```	
or

b)***Serve the prediction script as a service***

1-Start the Flask application using Gunicorn:

	```bash
	gunicorn predict:app --bind 0.0.0.0:8080
	``` 
	
2-Use curl to send a request to the service with the path or URL of an image:

	```bash	
	curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" \
	-d '{ "url":  "http://bit.ly/mlbookcamp-pants"}'
	``` 

The response should be something like:

	```bash	   
	{"Jackets":0.11359871178865433,"Shorts":0.028773579746484756,"Tops":0.02065521851181984,"Track Pants":0.5611394643783569,"Trousers":0.24156157672405243}
	```

---


## Project Structure

The relevant folders and files in this project are the ones listed below:

```
fashion-classification/
│
├── dataset/		# Contains the small version of the dataset and some images for testing
│   ├── fashion-product-images-small/	# Folder with the small images dataset downloaded from kaggle
│		├── images		# Images as .jpg
│		├── styles.csv		# Csv with all the information about the images
│   ├── test-images/		# Folder with large images for test	
├── fashion-classifier/		# Folder created with AWS SAM CLI for the deployment to AWS Lambda
│   ├── app/		# Folder with all the necessary files for deployment
│		├── Dockerfile		# Docker configuration
│		├── app.py			# Script with lambda function
│		├── classification_model.tflite		# Trained Tflite model
│		├── class_indices.json				# Json with the name and indices of the classes used to train the model
│		├── requirements.txt				# List of dependencies that must be installed in the Docker container		
├── models/		# Models created during evaluation and training of different CNN
├── notebooks/		# Jupyter notebooks for analysis
├── Pipfile		# Project top-level requirements (to be used with pipenv)
├── Pipfile.lock	# Required dependencies with specific versions (to be used with pipenv)
├── README.md		# Project documentation
├── scripts		# Scripts used for predictions and training
├── video/		# Folder that shows the lambda function being tested in AWS Lambda
└── .gitignore		# Git ignore file
```

---

## Jupyter notebooks

The Jupyter notebooks for this project were run locally on a machine with a GPU, where TensorFlow was installed. The explanation on how to set up the environment for TensorFlow is outside the scope of this document, but without it, these notebooks cannot be run.

The Jupyter notebooks in this project were used for Exploratory Data Analysis, Training, and Inference. Below is a description of each notebook:

-**notebook_1_eda_small_images.ipynb**

This notebook was used for the Exploratory Data Analysis of the [Small Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and to train several neural networks while tuning their parameters.

The final result in this notebook is the selection of the model to be trained on the larger dataset


-**notebook2_training_large_dataset.ipynb**

This notebook was used primarily to train the model selected in the notebook1 on the [Large Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

-**predict.ipynb**

This notebook was used to make predictions using the model trained in the notebook 2, leveraging Tensorflow for inference

-**predict_tflite.ipynb**

This notebook was used to create a TensorFlow Lite (TFLite) version of the model and use it to make predictions via the TFLite runtime

---

## Scripts

There are other files on the scripts folder besides the predict.py script mentioned previously. The one that is worth mention is the training.py script that was created from the notebook 2 and that can be used to train the model on the large Kagglehub dataset

---

## AWS Lambda

For the deployment of the project to AWS Lambda, I used the AWS SAM CLI to create a baseline and then adjusted the files as needed. The function was deployed successfully, and a video demonstrating a simple test of this function can be found in the /video folder.

