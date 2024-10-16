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
import db_utils # Importing utility functions for database operations

def ingest_data(data_path, mongo_host, mongo_port, mongo_database, collection_name):
    data = pd.read_csv(data_path)  # Read data from CSV file

    # Select relevant columns for MongoDB insertion
    mongo_data = data[['Tweet ID','entity','sentiment','Tweet content']]
    
    # Connect to MongoDB (without username or password)
    mongo_db = db_utils.connect_to_mongodb(mongo_host, mongo_port, mongo_database)
    
    # Insert data into MongoDB
    db_utils.insert_data_to_mongodb(mongo_data, collection_name, mongo_db)
