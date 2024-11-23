# Transfer Learning 

This project leverages transfer learning techniques for two key tasks: **mask detection in public spaces** and **sentiment analysis of Twitter data**. It employs state-of-the-art models such as BERT for NLP to make real-time predictions and analyze customer sentiment, aiding both public safety and digital marketing efforts.
This project also ensures mask-wearing compliance in public spaces using Computer Vision. It employs YOLOv8 for real-time mask detection and RCNN as an alternative for higher accuracy. The project showcases DIY Transfer Learning, adapting pre-trained models to detect masks with minimal data and resources.

## Table of Contents

1. [Problem Definition](#problem-definition)
   - [Public Health and Safety (Mask Detection)](#public-health-and-safety-mask-detection)
   - [Digital Marketing (Sentiment Analysis)](#digital-marketing-sentiment-analysis)
  
2. [Data Overview](#data-overview)
   - [Twitter Sentiment Analysis Dataset](#twitter-sentiment-analysis-dataset)
   - [Face Mask Detection Dataset](#face-mask-detection-dataset)

3. [Directory Structure](#directory-structure)

4. [Program Flow](#program-flow)
   - [db_utils](#db_utils)
   - [Data Ingestion](#data-ingestion)
   - [Data Preprocessing](#data-preprocessing)
   - [Data Splitting](#data-splitting)
   - [Model Training](#model-training)
   - [Web Application](#web-application)

5. [Transfer Learning Algorithms](#transfer-learning-algorithms)
   - [BERT (NLP)](#bert-nlp)
   - [YOLOv8 (Computer Vision)](#yolov8-computer-vision)
   - [FastRCNN (Neural Networks)](#fastrcnn-neural-networks)

6. [Steps to Run](#steps-to-run)
   - [Database Setup (MongoDB)](#database-setup-mongodb)
   - [Install Dependencies](#install-dependencies)
   - [Run Streamlit App](#run-streamlit-app)

7. [Error Handling in the Streamlit App](#error-handling-in-the-streamlit-app)

## Problem Definition

### Public Health and Safety (Mask Detection)
The business is focused on ensuring mask-wearing compliance in public spaces (e.g., airports and shopping malls). This solution uses **Computer Vision** models to detect individuals wearing masks or not, based on an image dataset.

### Digital Marketing (Sentiment Analysis)
The business also focuses on **sentiment analysis** of customer feedback gathered from social media platforms, specifically **Twitter**. The aim is to refine marketing strategies based on real-time sentiment analysis of tweets, helping the business respond to consumer feedback effectively.

## Data Overview

### 1. **Twitter Sentiment Analysis Dataset**
- **Dataset Link to download:** https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data
- **Columns:** tweetID, entity, sentiment (positive, neutral, negative), tweet_content.
- **Size:** 75,682 rows.
- **Use Case:** The dataset is used for sentiment analysis, where the goal is to classify tweets as positive, neutral, or negative based on the content.
- **Challenge:** The model needs to handle imbalanced sentiments across different entities for better generalization.

### 2. **Face Mask Detection Dataset**
- **Dataset Link:** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **Data Type:** Images stored in one folder and corresponding annotations in XML files stored in another folder.
- **Size:** 853 images.
- **Use Case:** The dataset is used to detect whether individuals in images are wearing face masks or not.
- **Challenge:** The model's accuracy can improve by including diverse lighting conditions and angles.

## Directory Structure

•⁠  ⁠*Code:* Contains all scripts for data ingestion, transformation, model training, evaluation, and web application. - *Data:* Contains Twitter data, face mask data, and data split during training. 
•⁠  ⁠*runs*: Stores the results of the - Computer Vision model on the face mask dataset.

## Program Flow

1.⁠ ⁠*db_utils:* Utility functions for MongoDB database connection, table creation, and data insertion.  
   [db_utils.py]

2.⁠ ⁠*Data Ingestion:* Ingests Twitter data from a CSV file, preprocesses it, and stores it in MongoDB.  
   [ingest.py]

3.⁠ ⁠*Data Preprocessing:* Preprocesses the data by scaling numerical columns, encoding categorical columns, and extracting date components.  
   [preprocess.py]

4.⁠ ⁠*Data Splitting:* Splits preprocessed data into training, validation, and super validation sets, and stores them in MongoDB.  
   [split.py]

5.⁠ ⁠*Model Training:* Trains models for *Neural Networks (NN), **Natural Language Processing (NLP), and **Computer Vision (CV)* tasks, and stores them in MongoDB.  
   [nn.py], [nlp.py], [cv.py]


6.⁠ ⁠*Web Application:* A Streamlit web app to train, evaluate, and classify Twitter sentiment or detect face masks using three different transfer learning models.  
   [app.py]

In addition to the Streamlit app for training and evaluating models, detailed outputs and preprocessing steps for YOLOv8 and FastRCNN are available in the corresponding Jupyter Notebook files. These notebooks provide a deeper understanding of the image preprocessing, training process, and detailed results (including accuracy and visualizations of the detection).

- **YOLOv8 Mask Detection Notebook**: This notebook contains detailed steps for image preprocessing, model training, and visual results of mask detection using YOLOv8.
  - File: [Code/cv.ipynb]
  - **Key Sections**:
    - Image preprocessing techniques used for mask detection
    - Detailed performance metrics and output visualizations
    - Accuracy results

- **FastRCNN Mask Detection Notebook**: This notebook provides a comprehensive breakdown of FastRCNN for mask detection, including preprocessing and visual analysis of detection results.
  - File: [Code/nn.ipynb]
  - **Key Sections**:
    - Image preprocessing steps tailored for FastRCNN
    - Visual representation of detection outputs
    - Model evaluation metrics and accuracy

To better understand the inner workings of the models and how they perform on the dataset, users are encouraged to review these notebooks for detailed image-level results and explanations.

**Note**: The **NLP model** has been trained for only 1 epoch in the app for faster training. To achieve better accuracy, users are encouraged to increase the number of epochs.


## Transfer Learning Algorithms

•⁠  ⁠*BERT (Bidirectional Encoder Representations from Transformers) - NLP*
  - BERT is a pre-trained transformer model used for understanding the context of words in text. It excels in tasks like text classification, sentiment analysis, and named entity recognition. Fine-tuning BERT helps reduce training time and improve model accuracy when labeled data is limited.

•⁠  ⁠*YOLOv8 (You Only Look Once) - Computer Vision*
  - YOLOv8 is a high-performance object detection model known for its speed and accuracy. It is particularly suitable for real-time applications like traffic sign recognition and face mask detection. By applying transfer learning, YOLOv8 can be fine-tuned on specific datasets for custom object detection tasks.

•⁠  ⁠*FastRCNN - Neural Networks*
  - FastRCNN is a deep learning model designed for efficient object detection tasks. It integrates both region proposal generation and classification into a unified framework. Transfer learning with FastRCNN allows for quick adaptation to new domains, enhancing detection accuracy and reducing training time.


## Steps to Run
1. **Ensure the databases (MongoDB) are running:**

    - **MongoDB Setup using MongoDB Compass:**
     1. Install MongoDB: Follow the installation guide for your operating system from the official [MongoDB Documentation](https://docs.mongodb.com/manual/installation/).
     2. Open **MongoDB Compass** and connect to your MongoDB instance.
     3. Create a database named **"1"** in MongoDB Compass.
     4. `x_train` and other collections(like `x_val`, `x_superval`, `y_train` etc.) will get stored in the database **"1"**.

     Here are the CLI commands for setting up MongoDB without using MongoDB Compass:
      ### MongoDB Compass Setup (CLI)
      1. FOR WINDOWS:
      Install MongoDB
      - Download and install MongoDB from the official MongoDB website:
      [MongoDB Downloads](https://www.mongodb.com/try/download/community)
      - Follow the installation guide for Windows.
      FOR MACOS:
      - Tap the MongoDB formula and install it using Homebrew:
      `brew tap mongodb/brew`
      `brew install mongodb-community@5.0`
      2. After installation, open Command Prompt as an administrator and Start the MongoDB service: `net start MongoDB`
      (FOR MACOS: `brew services start mongodb/brew/mongodb-community`)
      3. Open a new Command Prompt window and run `mongo`
      4. Switch to '1' database: `use 1`


2. **Install the necessary packages:** `pip install -r requirements.txt`

3.	**Run the Streamlit web application:** `streamlit run Code/app.py`

## Error Handling in the Streamlit App
When running the Streamlit app, the application will check for required files before processing any data. If any of these files are missing, the app will display an error message. For example:
1. If the X_test file is not found, an error like "X_test file not found" will be shown.
2. If the model file is missing, you will see an error message like "Model not found".

To resolve these errors, the user needs to click on the Train Model button. This step ensures that all necessary files are created, and the error messages will no longer appear.
