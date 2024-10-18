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
import streamlit as st

def ingest_data(data_path, mongodb_host, mongodb_port, mongodb_db,mongo_collection):

    # Connect to MongoDB
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    collection = db[mongo_collection]
    st.write(f"Provided file path: {data_path}")
     # Check if the file exists
    if not os.path.exists(data_path):
        st.write("File does not exist at the provided path.")
        return
        
    # # Read and insert CSV data with utf-8 encoding
    if not data_path.lower().endswith('.csv'):
        st.write("The file is not a CSV file.")
        return
    with open(data_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            data = list(reader)
            if data:  # Insert only if data is not empty
                collection.insert_many(data)
                st.write(f"Inserted {len(data)} records into {mongo_collection}.")
            else:
                st.write("No data found in the CSV file.")
   

