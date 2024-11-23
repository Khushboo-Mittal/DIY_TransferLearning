# Developer details: 
    # Name: Tanisha Priya
    # Role: Architect

  # Version:
        # Version: V 1.0 (24 October 2024)
            # Developers: Tanisha Priya
            # Unit test: Pass
            # Integration test: Pass
     
# Description: This code snippet uses YoloV8 Model to train, evaluate, and predict whether 
# a person is wearing a mask, not wearing a mask, or is wearing in an incorrect manner.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.10.8
            # Streamlit 1.22.0

# Import necessary libraries
import streamlit as st
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from ultralytics import YOLO  # YOLOv8 library for object detection
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For heatmap visualization of confusion matrix


# Data Preprocessing Functions
def create_folder_structure(base_path):
    """
    This function creates the necessary folder structure for training, validation, testing,
    and supervalidation datasets.
    """
    folders = [
        'data/train/images', 'data/train/labels',
        'data/val/images', 'data/val/labels',
        'data/test/images', 'data/test/labels',
        'data/superval/images', 'data/superval/labels'
    ]
    # Creating the folder structure if not already present
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)


def convert_bbox(size, box):
    """
    This function converts the bounding box coordinates into a normalized form,
    relative to the image size (width and height).
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0  # Calculate x center of bounding box
    y = (box[1] + box[3]) / 2.0  # Calculate y center of bounding box
    w = box[2] - box[0]  # Width of bounding box
    h = box[3] - box[1]  # Height of bounding box
    # Return the normalized bounding box
    return (x * dw, y * dh, w * dw, h * dh)


def convert_annotation(xml_path, output_path, classes):
    """
    This function converts the XML annotations (from the dataset) into a format compatible
    with YOLO (a text file with the class ID and bounding box coordinates).
    """
    # Parse the XML file to extract annotation details
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)  # Width of the image
    h = int(size.find('height').text)  # Height of the image

    # Open output file to write the converted annotations
    with open(output_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text  # Class name (e.g., 'mask', 'no_mask')
            if cls not in classes:  # Skip class if it's not in the list of classes
                continue
            cls_id = classes.index(cls)  # Get the index of the class
            xmlbox = obj.find('bndbox')  # Bounding box coordinates
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))  # Box coordinates
            bb = convert_bbox((w, h), b)  # Convert to normalized coordinates
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")  # Write to output file


def process_images(image_files, image_folder, annotation_folder, output_path, split, classes):
    """
    This function processes the images and their corresponding annotations, and saves them into
    the appropriate folder for training, validation, test, or supervalidation.
    """
    for file in image_files:
        # Process the image
        src_img = os.path.join(image_folder, file)
        dst_img = os.path.join(output_path, f'data/{split}/images', file)
        shutil.copy(src_img, dst_img)  # Copy image to the correct folder

        # Process the annotation
        xml_file = os.path.splitext(file)[0] + '.xml'  # Get the corresponding XML file
        src_xml = os.path.join(annotation_folder, xml_file)
        dst_txt = os.path.join(output_path, f'data/{split}/labels', os.path.splitext(file)[0] + '.txt')
        convert_annotation(src_xml, dst_txt, classes)  # Convert and save annotations


def split_and_process_dataset(image_files, image_folder, annotation_folder, output_path, classes):
    """
    This function splits the dataset into training, validation, test, and supervalidation sets.
    It also processes the images and annotations for each split.
    """
    # Split the data into train, validation, and test sets
    train_val, test = train_test_split(image_files, test_size=0.2, random_state=42)  # 80% train, 20% test
    train, val_superval = train_test_split(train_val, test_size=0.25, random_state=42)  # 20% for val and superval combined
    val, superval = train_test_split(val_superval, test_size=0.5, random_state=42)  # 50% val, 50% superval

    # Process images and annotations for each dataset split
    for split, files in [('train', train), ('val', val), ('test', test), ('superval', superval)]:
        process_images(files, image_folder, annotation_folder, output_path, split, classes)


def create_yaml(output_path, classes):
    """
    This function generates a YAML configuration file, which will be used by YOLOv8 during training.
    It contains the paths to the image folders and the class names.
    """
    yaml_content = {
        'train': f'{os.getcwd()}/Data/train/images',
        'val': f'{os.getcwd()}/Data/val/images',
        'test': f'{os.getcwd()}/Data/test/images',
        'superval': f'{os.getcwd()}/Data/superval/images',
        'nc': len(classes),  # Number of classes
        'names': classes  # Class names
    }

    # Save the YAML content to a file
    with open(os.path.join(output_path, 'data', 'data.yaml'), 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)


# Function to train the YOLOv8 model with only model_path as parameter
def train_yolov8_model(model_path, epochs=1, batch_size=32, img_size=640):
    """
    This function trains the YOLOv8 model using the dataset specified in the YAML file.
    After training, it evaluates the model and saves the trained model.
    """
    # Load necessary paths and data
    data_yaml = f'{os.getcwd()}/Data/data.yaml'  # Path to the YAML file

    # Initialize the YOLOv8 model with the small version (yolov8s.pt)
    model = YOLO('yolov8s.pt')

    # Start training the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        plots=True,
    )

    # Evaluate the model after training using the test data specified in the YAML file
    results = model.val(data=data_yaml)

    # Extract evaluation metrics
    accuracy = results.metrics['accuracy']
    precision = results.metrics['precision']
    recall = results.metrics['recall']
    f1 = results.metrics['f1']
    conf_matrix = results.metrics['confusion_matrix']

    # Visualize the evaluation metrics
    visualize_evaluation_metrics(accuracy, precision, recall, f1, conf_matrix)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save the trained model
    save_model(model, model_path)

    return accuracy, precision, recall, f1


# Function to evaluate the YOLOv8 model using the test set
def evaluate_yolov8_model(model, data_yaml):
    """
    This function evaluates the YOLOv8 model using the test data and returns performance metrics.
    """
    results = model.val(data=data_yaml)  # Evaluate the model

    # Extract performance metrics
    accuracy = results['metrics']['accuracy']
    precision = results['metrics']['precision']
    recall = results['metrics']['recall']
    f1 = results['metrics']['f1']
    conf_matrix = results['confusion_matrix']

    return accuracy, precision, recall, f1, conf_matrix


# Function to save the trained model
def save_model(model, model_path):
    """
    This function saves the trained model to the specified path.
    """
    model.save(model_path)


# Visualization function for evaluation metrics and confusion matrix

def visualize_evaluation_metrics(accuracy, precision, recall, f1, conf_matrix):
    # Bar plot for Accuracy, Precision, Recall, F1-Score
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metric_values, color=['skyblue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.show()

    # Confusion Matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
