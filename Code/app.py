# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harshita, prachi, khushboo, tanisha
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita, prachi, khushboo, tanisha
            # Unit test: Pass
            # Integration test: Pass
     
     # Description: This code snippet creates a web app to train, evaluate, and predict if credit card is fraudulent according to Transaction behaviour using
    # three different Outlier Detection models (Unsupervised Learning): IsolationForest, LocalOutlierFactor, One-Class SVM.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0
            
            
#to run: streamlit run Code/app.py


import streamlit as st  # Used for creating the web app
import datetime  # Used for setting default value in streamlit form

# Importing the .py helper files
from ingest import ingest_data  # Importing the ingest_data function from ingest.py
from preprocess import load_and_preprocess_data  # Importing the load_and_preprocess_data function from preprocess.py
from split import split_data_and_store  # Importing the split_data function from split.py
from nn import train_model_nn as train_model_nn # Importing the train_model function from model_training.py
from cv import train_yolov8_model as train_model_cv # Importing the train_model function from model_training.py
# from nlp import train_model as train_model_nlp 
from model_eval import evaluate_model  # Importing the evaluate_model function from model_eval.py
from model_predict import predict_output  # Importing the predict_output function from model_predict.py
from db_utils import load_FaceMask_data_from_mongodb as load_data_from_mongodb


# Setting the page configuration for the web app
st.set_page_config(page_title="Credit Card Fraud Prediction", page_icon=":chart_with_upwards_trend:", layout="centered")

# Adding a heading to the web app
st.markdown("<h1 style='text-align: center; color: white;'>Credit Card Fraud Prediction </h1>", unsafe_allow_html=True)
st.divider()

# Declaring session states(streamlit variables) for saving the path throught page reloads
# This is how we declare session state variables in streamlit.

# MongoDB
if "mongodb_host" not in st.session_state:
    st.session_state.mongodb_host = "localhost"
    
if "mongodb_port" not in st.session_state:
    st.session_state.mongodb_port = 27017
    
if "mongodb_db" not in st.session_state:
    st.session_state.mongodb_db = "1"

# Paths
if "facemask_data_path" not in st.session_state:
    st.session_state.facemask_data_path = "Data/Master/FaceMaskData/images"
    
if "tweet_data_path" not in st.session_state:
    st.session_state.tweet_data_path = "Data/Master/TwitterData/twitter_data.csv"
    
if "nn_model_path" not in st.session_state:
    st.session_state.nn_model_path = "nn_model.pkl"
    
if "cv_model_path" not in st.session_state:
    st.session_state.cv_model_path = "cv_model.pkl"
    
if "nlp_model_path" not in st.session_state:
    st.session_state.nlp_model_path = "nlp_model.pkl"

# Creating tabs for the web app.
tab1, tab2, tab3, tab4 = st.tabs(["Model Config","Model Training","Model Evaluation", "Model Prediction"])

# Tab for Model Config
with tab1:
    st.subheader("Model Configuration")
    st.write("This is where you can configure the model.")
    st.divider()
    
    with st.form(key="Config Form"):
        tab_mongodb, tab_paths = st.tabs(["MongoDB", "Paths"])
        
        # Tab for MongoDB Configuration
        with tab_mongodb:
            st.markdown("<h2 style='text-align: center; color: white;'>MongoDB Configuration</h2>", unsafe_allow_html=True)
            st.write(" ")
            
            st.write("Enter MongoDB Configuration Details:")
            st.write(" ")
            
            st.session_state.mongodb_host = st.text_input("MongoDB Host", st.session_state.mongodb_host)
            st.session_state.mongodb_port = st.number_input("Port", st.session_state.mongodb_port)
            st.session_state.mongodb_db = st.text_input("DB", st.session_state.mongodb_db)
        
        # Tab for Paths Configuration
        with tab_paths:
            st.markdown("<h2 style='text-align: center; color: white;'>Paths Configuration</h2>", unsafe_allow_html=True)
            st.write(" ")
            
            st.write("Enter Path Configuration Details:")
            st.write(" ")
            
            st.session_state.facemask_data_path = st.text_input("FaceMaskData Data Path", st.session_state.facemask_data_path)
            st.session_state.tweet_data_path = st.text_input("Tweets Data Path", st.session_state.tweet_data_path)
            st.session_state.nn_model_path = st.text_input("NN Model Path", st.session_state.nn_model_path)
            st.session_state.cv_model_path = st.text_input("CV Model Path", st.session_state.cv_model_path)
            st.session_state.nlp_model_path = st.text_input("NLP Model Path", st.session_state.nlp_model_path)
            
        if st.form_submit_button(label="Save Config", use_container_width=True):
            st.write("Configurations Saved Successfully! ✅")

# Tab for Model Training
with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()
    
    # Training the Models
    selected_model = st.selectbox("Select Model", ["NLP", "NN", "CV"])
    if st.button("Train Model", use_container_width=True):  # Adding a button to trigger model training
        with st.spinner("Training model..."):  # Displaying a status message while training the model
            st.write("Ingesting data...")  # Displaying a message for data ingestion
            collection_name = "datacollectionname"
            if selected_model == "NLP":
                data_path = st.session_state.tweet_data_path
                collection_name = "tweet_data"
            else:
                data_path = st.session_state.facemask_data_path
                collection_name = "facemask_data"
            ingest_data(st.session_state.master_data_path, st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db,data_path,collection_name)  # Calling the ingest_data function
            st.write("Data Ingested Successfully! ✅")  # Displaying a success message
            
            if collection_name=="tweet_data":
                st.write("Preprocessing data...")  # Displaying a message for data preprocessing
                data_postgres_processed= load_and_preprocess_data(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db)  # Calling the load_and_preprocess_data function
                st.write("Data Preprocessed Successfully! ✅")  # Displaying a success message
                st.write("Splitting data into train, test, validation, and super validation sets...")  # Displaying a message for data splitting
                split_data_and_store(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, data_postgres_processed) # Calling the split_data function
                st.write("Data Split Successfully! ✅")  # Displaying a success message
            elif collection_name=="facemask_data":
                data = load_data_from_mongodb(st.session_state.mongodb_db, collection_name)
                st.write("Splitting data into train, test, validation, and super validation sets...")  # Displaying a message for data splitting
                split_data_and_store(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, data) # Calling the split_data function
                st.write("Data Split Successfully! ✅")  # Displaying a success message
            
            st.write("Training model...")  # Displaying a message for model training
            
            # Choosing the model to train based on the user's selection
            if selected_model == "NN":
                # Calling the train_model function and storing the training accuracy and best hyperparameters
                accuracy = train_model_nn(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.nn_model_path)
            elif selected_model == "CV":
                accuracy = train_model_cv(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.cv_model_path)
            # elif selected_model == "NLP":
            #     silhouette_avg, best_params = train_model_nlp(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.oneclass_svm_path)
            st.write("Model Trained Successfully! ✅")  # Displaying a success message
        
        # Displaying the training accuracy
        st.success(f"{selected_model} Model Successfully trained with average silhouette score: {accuracy:.5f}")

# Tab for Model Evaluation
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the trained models")
    st.divider()
    
    # Displaying the metrics for the NN Model
    st.markdown("<h3 style='text-align: center; color: black;'>NN Model</h3>", unsafe_allow_html=True)
    st.divider()
    
    # Get the model test, validation, and super validation metrics
    nn_test_accuracy, nn_test_prec, nn_test_recall, nn_test_f1, nn_val_accuracy, nn_val_prec, nn_val_recall, nn_val_f1, nn_superval_accuracy, nn_superval_prec, nn_superval_recall, nn_superval_f1 = evaluate_model(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.nn_model_path)
    
    # Display model metrics in three columns
    nn_col1,nn_col2, nn_col3 = st.columns(3)
    
    # Helper function to center text vertically at the top using markdown
    def markdown_top_center(text):
        return f'<div style="display: flex; justify-content: center; align-items: flex-start; height: 100%;">{text}</div>'

    with nn_col1:
        # Displaying metrics for test, validation, and super validation sets
        st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accracy: {nn_test_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_test_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_test_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_test_f1), unsafe_allow_html=True)

    with nn_col2:
        st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {nn_val_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_val_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_val_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_val_f1), unsafe_allow_html=True)

    with nn_col3:
        st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {nn_superval_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_superval_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_superval_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(nn_superval_f1), unsafe_allow_html=True)
        
    st.divider()
    
    # Displaying the metrics for the CV Model
    st.markdown("<h3 style='text-align: center; color: black;'>CV Model</h3>", unsafe_allow_html=True)
    st.divider()

    # Get the model test, validation, and super validation metrics for CV
    cv_test_accuracy, cv_test_prec, cv_test_recall, cv_test_f1, cv_val_accuracy, cv_val_prec, cv_val_recall, cv_val_f1, cv_superval_accuracy, cv_superval_prec, cv_superval_recall, cv_superval_f1 = evaluate_model(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.cv_model_path)
    # Display model metrics in three columns
    cv_col1, cv_col2, cv_col3 = st.columns(3)

    # Display LOF model metrics using the same helper function to center text vertically at the top
    with cv_col1:
        # Displaying metrics for test, validation, and super validation sets
        st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accracy: {cv_test_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_test_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_test_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_test_f1), unsafe_allow_html=True)

    with cv_col2:
        st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {cv_val_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_val_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_val_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_val_f1), unsafe_allow_html=True)

    with cv_col3:
        st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {cv_superval_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_superval_prec), unsafe_allow_html=True)
        st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_superval_recall), unsafe_allow_html=True)
        st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(cv_superval_f1), unsafe_allow_html=True)
        

    st.divider()


        
    # # Displaying the metrics for the NLP Model
    # st.markdown("<h3 style='text-align: center; color: black;'>NLP Model</h3>", unsafe_allow_html=True)
    # st.divider()

    # # Get the model test, validation, and super validation metrics for NLP
    # nlp_test_accuracy, nlp_test_prec, nlp_test_recall, nlp_test_f1, nlp_val_accuracy, nlp_val_prec, nlp_val_recall, nlp_val_f1, nlp_superval_accuracy, nlp_superval_prec, nlp_superval_recall, nlp_superval_f1 = evaluate_model(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.nlp_model_path)
    
    # # Display model metrics in three columns for NLP
    # nlp_col1, nlp_col2, nlp_col3 = st.columns(3)
    
    # with nlp_col1:
    #     # Displaying metrics for test, validation, and super validation sets
    #     st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accracy: {nlp_test_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_test_prec), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_test_recall), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_test_f1), unsafe_allow_html=True)

    # with nlp_col2:
    #     st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {nlp_val_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_val_prec), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_val_recall), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_val_f1), unsafe_allow_html=True)

    # with nlp_col3:
    #     st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {nlp_superval_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_superval_prec), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_superval_recall), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(nlp_superval_f1), unsafe_allow_html=True)
        
    
    st.divider()

      
# Tab for Model Prediction
with tab4:
    
    st.subheader("Model Prediction")
    st.write("This is where you can predict.")
    st.divider()

    # Creating a form for user input
    with st.form(key="PredictionForm"): 
        
        selected_model = st.selectbox(label="Select Model",
                                      options=["NN", "CV", "NLP"])
        
        # Mapping model names to their respective paths
        model_path_mapping = {
            "NN": st.session_state.nn_model_path,
            "CV": st.session_state.cv_model_path,
            # "NLP": st.session_state.nlp_model_path
        }
        # File uploader for image upload
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        # The form always needs a submit button to trigger the form submission
        if st.form_submit_button(label="Predict", use_container_width=True):
            user_input = [uploaded_image, model_path_mapping[selected_model]]
            
            st.write(predict_output(*user_input))  # Calling the predict_output function with user input and displaying the output
