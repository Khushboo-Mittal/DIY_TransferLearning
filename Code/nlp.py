import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TFBertForSequenceClassification
from datasets import Dataset
import tensorflow as tf
from pymongo import MongoClient
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pickle  # For loading data
from db_utils import load_data_from_mongodb

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    with open(model_path, "wb") as f:
        # Serialize the model and save it to the file
        pickle.dump(model, f)
    
    
def evaluate_model(model, X_test_encoded, y_test):
    test_loss, test_accuracy = model.evaluate(
        [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
        y_test
    )
    
    # Get the predicted probabilities
    y_pred_probs = model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])

    # Convert the probabilities to predicted classes (assuming a binary classification, adjust for multi-class if needed)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1_score = f1_score(y_test, y_pred, average='weighted')
    
    return test_accuracy, precision, recall, f1_score, y_pred

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
        batch_size=32,
        epochs=3
    )
    
    test_accuracy, precision, recall, f1_score, pred = evaluate_model(model, X_test_encoded, y_test)
    
    save_model(model, model_path)
    
    return test_accuracy, precision, recall, f1_score, pred
    