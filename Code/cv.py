# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0



# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0

# Import necessary libraries
import pandas as pd
import numpy as np
import cv2  # For image processing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pymongo import MongoClient
import joblib  # For saving models
import pickle  # For loading data
import streamlit as st  # Ensure Streamlit is imported
from ultralytics import YOLO  # Ultralytics YOLOv8
from db_utils import load_FaceMask_data_from_mongodb as load_data

def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)  # Load image using OpenCV
        image = cv2.resize(image, (640, 640))  # Resize to 640x640 as YOLOv8 expects this input size
        images.append(image)
    return np.array(images)

def train_yolov8_model(mongodb_host, mongodb_port, mongodb_db, model_path):
    # Connect to the MongoDB database
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    X_train = load_data(db, 'x_train')
    # Preprocess the image dataset
    X_train = preprocess_images(X_train)

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model (smallest version)

    # Train the model
    results = model.train(data='/path/to/your/data.yaml', epochs=10)

    # Save the model as a .pt file (PyTorch format)
    model.save(model_path)
    
    # Save model weights as .pkl (serialize the model weights)
    model_weights = model.model.state_dict()  # Get model weights as a state dict
    with open(f'{model_path}.pkl', 'wb') as f:
        pickle.dump(model_weights, f)

    accuracy = evaluate_yolov8_model(model,X_test, y_test)
    return accuracy

def evaluate_yolov8_model(model, X_test, y_test):
    predictions = []
    for image in X_test:
        results = model.predict(source=image)  # Run the model prediction
        predicted_label = results[0].boxes.cls  # Extract the predicted class
        predictions.append(predicted_label)

    # Convert predictions to NumPy array for metric calculations
    y_pred = np.array(predictions)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)


    # Assuming you have y_test (true labels) and y_pred (predicted labels)
    # class_report = classification_report(y_test, y_pred, output_dict=True)

    # Extracting precision, recall, and F1-score
    # precision = class_report['weighted avg']['precision']
    # recall = class_report['weighted avg']['recall']
    # f1_score = class_report['weighted avg']['f1-score']

    return accuracy

