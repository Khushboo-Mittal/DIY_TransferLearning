# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harshita
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

    # If the path is a CSV file, read and insert CSV data
    if os.path.isfile(data_path) and data_path.lower().endswith('.csv'):
        with open(data_path, mode="r") as file:
            reader = csv.DictReader(file)
            data = list(reader)
            collection.insert_many(data)

