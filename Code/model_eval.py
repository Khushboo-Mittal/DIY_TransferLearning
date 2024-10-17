# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita 
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains utility functions to evaluate a model using test, validation,
    # and super validation data stored in a MongoDB database.
        # MongoDB: Yes
     
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5   
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

# Import necessary libraries
from pymongo import MongoClient # For connecting to MongoDB database
import pickle # For loading the model from a pickle file
import pandas as pd # For data manipulation
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # For model evaluation

# Load test, validation, and super validation data from MongoDB
def load_data_from_mongodb(db, collection_name):
    # Connect to MongoDB
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'data' in data:
        return pickle.loads(data['data'])  # Deserialize the pickled binary data
    return None

def evaluate_test_data(X_test, y_test, model):
    # Predict labels for the test set
    y_pred_test = model.predict(X_test)  # Adjust this depending on your model's framework
    
    # If the model outputs probabilities (e.g., for binary classification), convert them to class labels
    if y_pred_test.ndim > 1 and y_pred_test.shape[1] > 1:  # Check if output is multi-class
        y_pred_test_classes = np.argmax(y_pred_test, axis=1)  # Get class labels for multi-class
    else:
        y_pred_test_classes = (y_pred_test > 0.5).astype(int)  # Thresholding for binary classification
    
    # Calculate accuracy score for the test set
    test_accuracy = accuracy_score(y_test, y_pred_test_classes)
    
    # Calculate precision, recall, and F1 score for the test set
    test_precision = precision_score(y_test, y_pred_test_classes, average='weighted')
    test_recall = recall_score(y_test, y_pred_test_classes, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test_classes, average='weighted')
    
    # Return accuracy, precision, recall, and F1 score for the test set
    return (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    )

def evaluate_validation_data(X_val, y_val, model):
    # Predict labels for the validation set
    val_pred = model.predict(X_val)  # Adjust this based on your model's prediction method
    
    # Convert predictions to class labels if needed
    if val_pred.ndim > 1 and val_pred.shape[1] > 1:  # Check for multi-class
        val_pred_classes = np.argmax(val_pred, axis=1)
    else:
        val_pred_classes = (val_pred > 0.5).astype(int)  # Threshold for binary classification
    
    # Calculate accuracy, precision, recall, and F1 score
    val_accuracy = accuracy_score(y_val, val_pred_classes)
    val_precision = precision_score(y_val, val_pred_classes, average='weighted')
    val_recall = recall_score(y_val, val_pred_classes, average='weighted')
    val_f1 = f1_score(y_val, val_pred_classes, average='weighted')

    return val_accuracy, val_precision, val_recall, val_f1

def evaluate_supervalidation_data(X_superval, y_superval, model):
    # Predict labels for the supervalidation set
    superval_pred = model.predict(X_superval)  # Adjust this based on your model's prediction method
    
    # Convert predictions to class labels if needed
    if superval_pred.ndim > 1 and superval_pred.shape[1] > 1:  # Check for multi-class
        superval_pred_classes = np.argmax(superval_pred, axis=1)
    else:
        superval_pred_classes = (superval_pred > 0.5).astype(int)  # Threshold for binary classification
    
    # Calculate accuracy, precision, recall, and F1 score
    superval_accuracy = accuracy_score(y_superval, superval_pred_classes)
    superval_precision = precision_score(y_superval, superval_pred_classes, average='weighted')
    superval_recall = recall_score(y_superval, superval_pred_classes, average='weighted')
    superval_f1 = f1_score(y_superval, superval_pred_classes, average='weighted')

    return superval_accuracy, superval_precision, superval_recall, superval_f1

def evaluate_model(mongodb_host, mongodb_port, mongodb_db, model_path):
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    
    X_test = load_data_from_mongodb(db, 'x_test')
    Y_test = load_data_from_mongodb(db,'y_train')
    X_val = load_data_from_mongodb(db, 'x_val')
    Y_val = load_data_from_mongodb(db,'y_val')
    X_superval = load_data_from_mongodb(db, 'x_superval')
    Y_superval = load_data_from_mongodb(db,'y_superval')
    
    # Ensure column names are strings for consistency
    X_test = X_test.rename(str, axis="columns")
    Y_test = Y_test.rename(str, axis="columns")
    X_val = X_val.rename(str, axis="columns")
    Y_val = Y_val.rename(str, axis="columns")
    X_superval = X_superval.rename(str, axis="columns")
    Y_superval = Y_superval.rename(str, axis="columns")
    
    X_test = X_test.values
    Y_test = Y_test.values
    X_val = X_val.values
    Y_val = Y_val.values
    X_superval = X_superval.values
    Y_superval = Y_superval.values
    

    # Load the best model from the pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
   # Evaluate the model on test data
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_test_data(X_test, Y_test, model)

    # Evaluate the model on validation data
    val_accuracy, val_precision, val_recall, val_f1 = evaluate_validation_data(X_val, Y_val, model)

    # Evaluate the model on super validation data
    superval_accuracy, superval_precision, superval_recall, superval_f1 = evaluate_supervalidation_data(X_superval, Y_superval, model)

    
    # Return evaluation metrics for test, validation, and super validation data
    return (test_accuracy, test_precision, test_recall, test_f1,
            val_accuracy, val_precision, val_recall, val_f1,
            superval_accuracy, superval_precision, superval_recall, superval_f1)
