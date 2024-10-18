# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita and Prachi
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and Prachi
            # Unit test: Pass
            # Integration test: Pass
            
    # Description: This code snippet preprocesses input data for a machine learning model by scaling numerical
    # columns, encoding categorical columns, and extracting date components for further analysis.
        # PostgreSQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5     
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                             # For data manipulation
import pickle                                                   # For loading the model from a pickle file
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing input data
import spacy
def preprocess_input_data(text):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm") 

    def preprocess(text):
        doc = nlp(text)
        filtered_tokens = []
        for token in doc:
            # Skip stopwords and punctuation
            if not token.is_stop and not token.is_punct:
                filtered_tokens.append(token.lemma_)  # Lemmatize the token
                
        return " ".join(filtered_tokens)  
    data = preprocess(data)
    return data

def predict_output(image,tweet_content, model_path):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Preprocess input data
    data = preprocess_input_data(tweet_content)
    
    prediction = model.predict(data.values)  # Make a prediction (assume only one prediction is made)
    str = ""
    # if prediction==-1:
    #     str += "It's fradulent"
    # else:
    #     str += "It's not fradulent"
        
    return f"Model Prediction: {prediction}, thus {str}"  # Return the prediction
