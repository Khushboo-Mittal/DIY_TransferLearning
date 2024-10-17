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

def load_data_from_mongodb(db, collection_name):
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'data' in data:
        return pickle.loads(data['data'])  # Deserialize the pickled binary data
    return None

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

    # Load the image paths and labels from MongoDB
    image_paths = load_data_from_mongodb(db, 'image_paths')
    labels = load_data_from_mongodb(db, 'labels')

    # Preprocess the image dataset
    X_train = preprocess_images(image_paths)

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

    print('Model training completed successfully!')

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
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Extracting precision, recall, and F1-score
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']
    f1_score = class_report['weighted avg']['f1-score']

    # Printing the results
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")

    return accuracy,precision,f1_score

def main(mongodb_host, mongodb_port, mongodb_db, model_path):
    # Connect to MongoDB and load db reference
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]

    # Train the YOLOv8 model
    train_yolov8_model(mongodb_host, mongodb_port, mongodb_db, model_path)

    # Load test data
    X_test = preprocess_images(load_data_from_mongodb(db, 'test_image_paths'))
    y_test = load_data_from_mongodb(db, 'test_labels')

    # Load the saved YOLOv8 model
    model = YOLO(model_path)

    # Evaluate the model
    accuracy, conf_matrix, class_report = evaluate_yolov8_model(model, X_test, y_test)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')

    main(mongodb_host, mongodb_port, mongodb_db, model_path)
