# Developer details: 
        # Name: Prachi Tavse
        # Role: Architects
    # Version:
        # Version: V 1.0 (19 October 2024)
            # Developers: Prachi Tavse
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet creates the NLP Model to train, evaluate, and classify the 
    # tweet sentiment whether if its positive, negative or neutral

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import Dataset
import tensorflow as tf
from pymongo import MongoClient
from sklearn.metrics import precision_score, recall_score, f1_score
from db_utils import load_data_from_mongodb

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    model.save(model_path)

    
    
def evaluate_model(model, X_test_encoded, y_test):
    est_loss, test_accuracy = model.evaluate(
        [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
        y_test
    )
    
    # Get the predicted probabilities
    logits = model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']]).logits
    
    y_pred_probs = tf.nn.softmax(logits, axis=-1).numpy()

    # If it's binary classification, use a threshold of 0.5 on the positive class probability
    if y_pred_probs.shape[1] == 2:
        y_pred = (y_pred_probs[:, 1] > 0.5).astype(int)
    else:
        # For multi-class classification, take the argmax of the probabilities
        y_pred = np.argmax(y_pred_probs, axis=1)
        
    y_test = np.array(y_test)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return test_accuracy, precision, recall, f1,y_pred

def train_model(mongodb_host, mongodb_port, mongodb_db, model_path):

    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    # Load the training data from MongoDB
    X_train = load_data_from_mongodb(db,'x_train')
    # X_train = X_train.values
    X_test = load_data_from_mongodb(db,'x_test')
    # X_test = X_test.values
    X_val = load_data_from_mongodb(db,'x_val')
    # X_val = X_val.values
    
    y_train = load_data_from_mongodb(db,'y_train')
    # y_train = y_train.values
    y_test = load_data_from_mongodb(db,'y_test')
    # y_test = y_test.values
    y_val = load_data_from_mongodb(db,'y_val')
    # y_val = y_val.values
    
    xlist = [X_train, X_test, X_val]
    X = pd.concat(xlist)
    
    pdList = [y_train, y_test, y_val]  # List of your dataframes
    y = pd.concat(pdList)
    
    #Tokenize and encode the data using the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len= 128
    # Tokenize and encode the sentences
    X_train_encoded = tokenizer.batch_encode_plus(X.tolist(),
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_val_encoded = tokenizer.batch_encode_plus(X_val.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_test_encoded = tokenizer.batch_encode_plus(X_test.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

    # Intialize the model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Compile the model with an appropriate optimizer, loss function, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    history = model.fit(
        [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
        y,
        validation_data=(
        [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],y_val),
        batch_size=16,
        epochs=1
    )
    
    test_accuracy, precision, recall, f1, pred = evaluate_model(model, X_test_encoded, y_test)
    
    save_model(model, model_path)
    
    return test_accuracy, precision, recall, f1
    