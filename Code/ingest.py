# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harshita and Prachi
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and Prachi
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet ingests transaction data from a CSV file, preprocesses it, and stores it in
    # PostgreSQL database.
        # PostgreSQL: Yes 

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2

import pandas as pd # Importing pandas for data manipulation
import csv
import os
from pymongo import MongoClient

def ingest_data(data_path, mongodb_host, mongodb_port, mongodb_db,mongo_collection):

    # Connect to MongoDB
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    collection = db[mongo_collection]

    # Check if the data_path is a directory for images and annotations
    images_path = os.path.join(data_path, "images")
    annotations_path = os.path.join(data_path, "annotations")

    # Process image files
    if os.path.isdir(images_path):
        for filename in os.listdir(images_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add other image formats as needed
                image_path = os.path.join(images_path, filename)
                # Insert the image path into MongoDB
                collection.insert_one({"image_path": image_path})

    # Process annotation files
    if os.path.isdir(annotations_path):
        for filename in os.listdir(annotations_path):
            if filename.lower().endswith('.xml'):  # Checking for XML files
                annotation_path = os.path.join(annotations_path, filename)
                # Insert the annotation path into MongoDB
                collection.insert_one({"annotation_path": annotation_path})

    # If the path is a CSV file, read and insert CSV data
    elif os.path.isfile(data_path) and data_path.lower().endswith('.csv'):
        with open(data_path, mode="r") as file:
            reader = csv.DictReader(file)
            data = list(reader)
            collection.insert_many(data)

