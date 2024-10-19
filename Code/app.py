# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harshita, Prachi, Khushboo and Tanisha
        # Role: Architects
    # Version:
        # Version: V 1.0 (19 October 2024)
            # Developers: Harshita, Prachi, Khushboo and Tanisha
            # Unit test: Pass
            # Integration test: Pass
     
     # Description: This code snippet creates a web app to train, evaluate, and classifiy twitter tweet sentiment and detect face mask using
    # three different Deep Learning models (Transfer Learning): NN, NLP, abd CV.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0
            
            
#to run: streamlit run Code/app.py


import streamlit as st  # Used for creating the web app

from ingest import ingest_data  # Importing the ingest_data function from ingest.py
from preprocess import load_and_preprocess_data  # Importing the load_and_preprocess_data function from preprocess.py
from split import split_data_and_store  # Importing the split_data function from split.py
from nn3 import train_model as train_model_nn # Importing the train_model function from model_training.py
from cv import train_yolov8_model as train_model_cv # Importing the train_model function from model_training.py
from nlp import train_model as train_model_nlp 


# Setting the page configuration for the web app
st.set_page_config(page_title="Transfer Learning", page_icon=":chart_with_upwards_trend:", layout="centered")

# Adding a heading to the web app
st.markdown("<h1 style='text-align: center; color: white;'>Transfer Learning </h1>", unsafe_allow_html=True)
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
    st.session_state.facemask_data_path = "Data/FaceMaskData/images"
    
if "tweet_data_path" not in st.session_state:
    st.session_state.tweet_data_path = "Data/TwitterData/twitter_data.csv"
    
if "nn_model_path" not in st.session_state:
    st.session_state.nn_model_path = "nn_model.pt"
    
if "cv_model_path" not in st.session_state:
    st.session_state.cv_model_path = "yolov8.pt"
    
if "nlp_model_path" not in st.session_state:
    st.session_state.nlp_model_path = "nlp_model.h5"

#defining metrics here to store later    
nn_test_accuracy, nn_test_prec, nn_test_recall = 0, 0, 0
cv_test_accuracy,cv_test_prec,cv_test_recall =0,0,0
nlp_test_accuracy, nlp_test_prec, nlp_test_recall, nlp_test_f1 = 0, 0, 0, 0





# Creating tabs for the web app.
tab1, tab2, tab3 = st.tabs(["Model Config","Model Training","Model Evaluation"])

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
            if selected_model == "NLP":
                st.write("Ingesting data...")  # Displaying a message for data ingestion
                data_path = st.session_state.tweet_data_path
                collection_name = "tweet_data"
                ingest_data(st.session_state.tweet_data_path, st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db,collection_name)  # Calling the ingest_data function
                st.write("Data Ingested Successfully! ✅")  # Displaying a success message
                
                st.write("Preprocessing data...")  # Displaying a message for data preprocessing
                data_postgres_processed= load_and_preprocess_data(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db)  # Calling the load_and_preprocess_data function
                st.write("Data Preprocessed Successfully! ✅")  # Displaying a success message
                st.write("Splitting data into train, test, validation, and super validation sets...")  # Displaying a message for data splitting
                split_data_and_store(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, data_postgres_processed) # Calling the split_data function
                st.write("Data Split Successfully! ✅")  # Displaying a success message
                
            st.write("Training model...")  # Displaying a message for model training
            # Choosing the model to train based on the user's selection
            if selected_model == "NN":
                # Calling the train_model function and storing the training accuracy and best hyperparameters
                nn_test_accuracy,nn_test_prec,nn_test_recall = train_model_nn(st.session_state.nn_model_path)
                st.success(f"{selected_model} Model Successfully trained with accuracy score: {nn_test_accuracy:.5f}")
            elif selected_model == "CV":
                cv_test_accuracy,cv_test_prec,cv_test_recall = train_model_cv( st.session_state.cv_model_path)
                st.success(f"{selected_model} Model Successfully trained with accuracy score: {cv_test_accuracy:.5f}")
            if selected_model == "NLP":
                nlp_test_accuracy,nlp_test_prec,nlp_test_recall,nlp_test_f1 = train_model_nlp(st.session_state.mongodb_host, st.session_state.mongodb_port, st.session_state.mongodb_db, st.session_state.nlp_model_path)
                st.success(f"{selected_model} Model Successfully trained with accuracy score: {nlp_test_accuracy:.5f}")
            st.write("Model Trained Successfully! ✅")  # Displaying a success message
        
        # Displaying the training accuracy
        st.success(f"{selected_model} Model Successfully trained with accuracy score: {nlp_test_accuracy:.5f}")

# Tab for Model Evaluation
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the trained models")
    st.divider()
    
    # # Displaying the metrics for the NN Model
    st.markdown("<h3 style='text-align: center; color: black;'>NN Model</h3>", unsafe_allow_html=True)
    st.divider()
    
    # # Helper function to center text vertically at the top using markdown
    def markdown_top_center(text):
        return f'<div style="display: flex; justify-content: center; align-items: flex-start; height: 100%;">{text}</div>'

    # Displaying metrics for test, validation, and super validation sets
    st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(f"Accracy: {nn_test_accuracy:.5f}"), unsafe_allow_html=True)
    st.write(" ")
    st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(nn_test_prec), unsafe_allow_html=True)
    st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(nn_test_recall), unsafe_allow_html=True)
    st.divider()
    
    
    
    # Displaying the metrics for the CV Model
    st.markdown("<h3 style='text-align: center; color: black;'>CV Model</h3>", unsafe_allow_html=True)
    st.divider()

    # # Display LOF model metrics using the same helper function to center text vertically at the top
    st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(f"Accracy: {cv_test_accuracy:.5f}"), unsafe_allow_html=True)
    st.write(" ")
    st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(cv_test_prec), unsafe_allow_html=True)
    st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(cv_test_recall), unsafe_allow_html=True)
    st.divider()
    
    
        
    # Displaying the metrics for the NLP Model
    st.markdown("<h3 style='text-align: center; color: black;'>NLP Model</h3>", unsafe_allow_html=True)
    st.divider()
    # Displaying metrics for test, validation, and super validation sets
    st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(f"Accracy: {nlp_test_accuracy:.5f}"), unsafe_allow_html=True)
    st.write(" ")
    st.markdown(markdown_top_center("Precision:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(nlp_test_prec), unsafe_allow_html=True)
    st.markdown(markdown_top_center("Recall:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(nlp_test_recall), unsafe_allow_html=True)
    st.markdown(markdown_top_center("F1 Score:"), unsafe_allow_html=True)
    st.markdown(markdown_top_center(nlp_test_f1), unsafe_allow_html=True)
    st.divider()
            