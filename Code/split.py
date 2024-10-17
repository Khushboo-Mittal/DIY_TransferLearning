# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita and prachi
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and prachi
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains functions to split preprocessed data into test, validation,
    # and super validation and store it in a MongoDB database.
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5


# Importing necessary libraries
import pandas as pd                                      # For data manipulation
from sklearn.model_selection import train_test_split     # To split data into train, test, validation, and super validation sets
from pymongo import MongoClient                          # For using MongoDB as a cache to store the split data
import pickle                                            # For serializing and deserializing data for storage in MongoDB

def connect_to_mongodb(host, port, db_name):
    # Connect to MongoDB
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    return db

def save_preprocessed_data(preprocessed_data):
    # Save preprocessed data as CSV for inspection
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)  # Save to CSV

def split_data(preprocessed_data):
    """
    Split the data into training, testing, validation, and super validation sets.
    """
    X = preprocessed_data[:]  # Use the entire preprocessed data as features
    
    # Split the data into train, test, validation, and super validation sets
    X_train, X_temp = train_test_split(X, test_size=0.4, random_state=42)  # 60% train, 40% temp
    X_test, X_temp = train_test_split(X_temp, test_size=0.625, random_state=42)  # 0.625 * 0.4 = 0.25 for test
    X_val, X_superval = train_test_split(X_temp, test_size=0.5, random_state=42)  # 0.5 * 0.25 = 0.125 for validation and super validation
    return X_train, X_test, X_val, X_superval

def store_to_mongo(data, db, collection_name):
    """
    Store the data into a MongoDB collection.
    """
    collection = db[collection_name]  # Select the collection
    collection.insert_one({'data': data})  # Insert the data into the collection

def save_split_data(db, X_train, X_test, X_val, X_superval):
    """
    Store the split data (train, test, val, superval) into MongoDB.
    """
    store_to_mongo(pickle.dumps(X_train), db, 'x_train')
    store_to_mongo(pickle.dumps(X_test), db, 'x_test')
    store_to_mongo(pickle.dumps(X_val), db, 'x_val')
    store_to_mongo(pickle.dumps(X_superval), db, 'x_superval')

def load_data_from_mongo(db, collection_name):
    """
    Load the preprocessed data from MongoDB collection.
    """
    collection = db[collection_name]  # Access the collection
    data = collection.find_one()  # Get the first document
    return pickle.loads(data['data'])  # Deserialize and return data

def split_data_and_store(mongodb_host, mongodb_port, mongodb_db, collection_name):
    """
    Main function to load, preprocess, split, and store the data into MongoDB.
    """
    # Connect to MongoDB
    db = connect_to_mongodb(mongodb_host, mongodb_port, mongodb_db)
   
    # Load the preprocessed data from MongoDB
    preprocessed_data = load_data_from_mongo(db, collection_name)
    
    # Uncomment the below line to see how the merged processed data looks
    save_preprocessed_data(preprocessed_data)
    
    # Split data into train, test, validation, and super validation sets
    X_train, X_test, X_val, X_superval = split_data(preprocessed_data)
    
    # Save split data into MongoDB
    save_split_data(db, X_train, X_test, X_val, X_superval)
    
    print('Data preprocessed, and split successfully!')

# Example usage
if __name__ == '__main__':
    mongodb_host = 'localhost'
    mongodb_port = 27017
    mongodb_db = 'my_database'  # Replace with your MongoDB database name
    collection_name = 'preprocessed_data'  # Replace with the collection where the data is stored

    # Run the function to split data
    split_data_and_store(mongodb_host, mongodb_port, mongodb_db, collection_name)
