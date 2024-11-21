# Transfer Learning 

This project leverages transfer learning techniques for two key tasks: **mask detection in public spaces** and **sentiment analysis of Twitter data**. It employs state-of-the-art models such as BERT for NLP and YOLOv8 for Computer Vision to make real-time predictions and analyze customer sentiment, aiding both public safety and digital marketing efforts.

## Table of Contents
1. [Problem Definition]
   - [Public Health and Safety (Mask Detection)]
   - [Digital Marketing (Sentiment Analysis)]
2. [Data Overview]
   - [Twitter Sentiment Analysis Dataset]
   - [Face Mask Detection Dataset]
3. [Directory Structure]
4. [Program Flow]
   - [db_utils]
   - [Data Ingestion]
   - [Data Preprocessing]
   - [Data Splitting]
   - [Model Training]
   - [Web Application]
5. [Transfer Learning Algorithms]
   - [BERT (NLP)]
   - [YOLOv8 (CV)]
   - [FastRCNN (Neural Networks)]
6. [Steps to Run the Application]

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

## Transfer Learning Algorithms

•⁠  ⁠*BERT (Bidirectional Encoder Representations from Transformers) - NLP*
  - BERT is a pre-trained transformer model used for understanding the context of words in text. It excels in tasks like text classification, sentiment analysis, and named entity recognition. Fine-tuning BERT helps reduce training time and improve model accuracy when labeled data is limited.

•⁠  ⁠*YOLOv8 (You Only Look Once) - Computer Vision*
  - YOLOv8 is a high-performance object detection model known for its speed and accuracy. It is particularly suitable for real-time applications like traffic sign recognition and face mask detection. By applying transfer learning, YOLOv8 can be fine-tuned on specific datasets for custom object detection tasks.

•⁠  ⁠*FastRCNN - Neural Networks*
  - FastRCNN is a deep learning model designed for efficient object detection tasks. It integrates both region proposal generation and classification into a unified framework. Transfer learning with FastRCNN allows for quick adaptation to new domains, enhancing detection accuracy and reducing training time.

## Steps to Run the Application

1.⁠ ⁠Install the necessary packages: pip install -r requirements.txt
2.⁠ ⁠Run the Streamlit web application: streamlit run Code/app.py

