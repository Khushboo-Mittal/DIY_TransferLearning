# DIY - TRANSFER LEARNING
This branch contains tranfer learning algorithms.

## Transfer Learning Algorithms
### BERT (Bidirectional Encoder Representations from Transformers) - NLP
BERT is a pre-trained transformer model that excels at understanding the context of words in text by considering both their left and right surroundings. In transfer learning, BERT can be fine-tuned for a variety of NLP tasks, including text classification, sentiment analysis, and named entity recognition. Its ability to handle large text corpora and provide contextualized word embeddings makes it highly effective for tasks where labeled data is limited. Fine-tuning BERT significantly reduces training time and improves model accuracy for specific language tasks.

### YOLOv8 (You Only Look Once) - Computer Vision
YOLOv8 is a high-performance object detection model that offers an excellent balance between detection speed and accuracy. It is particularly suited for real-time applications like traffic sign recognition, face mask detection, and autonomous driving. By leveraging transfer learning, YOLOv8 can be fine-tuned on specific image datasets, allowing the model to adapt pre-trained weights for detecting custom objects. This results in faster training, reduced computational cost, and improved accuracy, even when working with smaller image datasets.

### FastRCNN - Neural Networks
FastRCNN is a deep learning model designed for object detection tasks, integrating both region proposal generation and classification into a single, more efficient framework. By applying transfer learning, FastRCNN can be fine-tuned for specific use cases like anomaly detection in time-series data or specialized object detection tasks. This model leverages pre-trained features to quickly adapt to new domains, improving detection accuracy and reducing the time needed for training on custom data.

## Problem Definition
The business operates in public health and safety, focusing on enforcing mask-wearing in public spaces. The goal is to use Computer Vision and Neural Network models, to detect individuals with or without masks in real-time based on an image dataset. These models ensure compliance in areas such as airports and shopping malls, enhancing public safety.

The business operates in the digital marketing sector, aiming to understand customer sentiment from social media platforms like Twitter. The goal is to use Natural Language Processing (NLP) for sentiment analysis of tweets, helping the company refine its marketing strategies and respond to consumer feedback in real time.

## Data Definition
1. **The Twitter sentiment analysis dataset contains the following columns:** tweetID, entity, sentiment (positive, neutral, negative), and tweet_content. This dataset consists of 75,682 rows, providing a substantial amount of data for training sentiment analysis models. However, the diversity of sentiments across entities may still affect the accuracy and F1-score metrics. Ensuring a balanced representation of sentiments will enhance the model's ability to generalize and accurately classify customer opinions.

2. **The face mask detection dataset includes images stored in one folder and corresponding annotations in XML files stored in another folder.** This dataset consists of 853 images, which is adequate for training mask detection models. However, the model's performance could still benefit from diverse lighting conditions and angles. Comprehensive annotations will help improve evaluation metrics like precision and recall, ensuring effective real-world mask detection.

## Directory Structure
**Code/:** Contains all the scripts for data ingestion, transformation, loading, evaluation, model training, inference, manual prediction, and web application.
**Data/:** Contains the face mask data and twitter data. It also contains the data splitted during cv model training
**runs/** Contains the result of cv model on face mask data

## Program Flow
1.	**db_utils:** This code snippet contains utility functions to connect to MongoDB database, create tables, and insert data into them.[`db_utils.py`]
2.	**Data Ingestion:** This code snippet ingests twitter data from CSV file, preprocesses it, and stores it in MongoDB database. [`ingest.py`]
3.	**Data Preprocessing:** This code snippet preprocesses input data for a machine learning model by scaling numerical columns, encoding categorical columns, and extracting date components for further analysis [`preprocess.py`]
4.	**Data Splitting:** This code snippet contains functions to split preprocessed data into test, validation, and super validation and store it in a MongoDB database. [`split.py`]
5.	**Model Training:** This is where NN, NLP and CV models, using the training data, are trained and stored in a MongoDB database. [`nn.py`, `nlp.py`, `cv.py`]
7.	**Model Prediction:** This code snippet predict twitter sentiment or detect face mask based on user input data.  [`model_predict.py`]
8.	**Web Application:** This code snippet creates a web app using Streamlit to train, evaluate, and classify twitter sentiment or detect face mask using three different transfer learning models: NN, NLP and CV. [`app.py`]

## Steps to run
1. Install the necessary packages: pip install -r requirements.txt
2. Run the Streamlit web application: streamlit run Code/app.py
