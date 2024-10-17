# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and Prachi
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains utility functions to connect to PostgreSQL database,
    # create tables, and insert data into them.
        # PostgreSQL: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5   
            # SQLAlchemy 2.0.31

from pymongo import MongoClient
import pandas as pd

def connect_to_mongodb(host, port, db_name):
    # Function to connect to MongoDB using the provided configuration
    client = MongoClient(host, port)  
    db = client[db_name]
    return db  # Return MongoDB database object

def insert_data_to_mongodb(data, collection_name, db):
    # Function to insert data into MongoDB collection
    collection = db[collection_name]
    
    # Check if the input data is a DataFrame (for CSV data)
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to dictionary format for MongoDB
        data_dict = data.to_dict(orient='records')
        # Insert data into MongoDB collection
        collection.insert_many(data_dict)
    
    # If data is not a DataFrame, assume it's a list of image paths
    elif isinstance(data, list):
        # Insert each image path into the MongoDB collection
        for image_path in data:
            collection.insert_one({"image_path": image_path})
            
# Load image and annotation paths from MongoDB
def load_FaceMask_data_from_mongodb(db, collection_name):
    collection = db[collection_name]
    image_paths, annotation_paths = [], []
    
    for document in collection.find():
        if 'image_path' in document:
            image_paths.append(document['image_path'])
        if 'annotation_path' in document:
            annotation_paths.append(document['annotation_path'])
    
    return image_paths, annotation_paths

