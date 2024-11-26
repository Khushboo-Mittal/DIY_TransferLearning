# TRANSFER LEARNING

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

5. [Transfer Learning Algorithms](#transfer-learning-algorithms)
   - [BERT (NLP)](#bert-nlp)
   - [YOLOv8 (Computer Vision)](#yolov8-computer-vision)
   - [FastRCNN (Neural Networks)](#fastrcnn-neural-networks)

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
- ⁠*Code:* Contains all models in ipynb files.
- *Data:* Contains Twitter data, face mask data, and data split during training.
- ⁠*runs*: Stores the results of the - Computer Vision model on the face mask dataset.

## Program Flow

- **BERT Sentiment Analysis Notebook**: This notebook provides a detailed walkthrough of using Transfer Learning with BERT for sentiment analysis on Twitter data, covering preprocessing, model training, and performance evaluation.
  - File: [Code/nlp.ipynb]
  - **Key Sections**:
    - Data preprocessing techniques for handling Twitter data
    - Implementation of Transfer Learning with BERT for sentiment analysis
    - Model performance metrics, including precision, recall and F1 scores

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

**Note**: The **models** has been trained for only few epochs for faster training. To achieve better accuracy, users are encouraged to increase the number of epochs.


## Transfer Learning Algorithms

•⁠  ⁠*BERT (Bidirectional Encoder Representations from Transformers) - NLP*
  - BERT is a pre-trained transformer model used for understanding the context of words in text. It excels in tasks like text classification, sentiment analysis, and named entity recognition. Fine-tuning BERT helps reduce training time and improve model accuracy when labeled data is limited.

•⁠  ⁠*YOLOv8 (You Only Look Once) - Computer Vision*
  - YOLOv8 is a high-performance object detection model known for its speed and accuracy. It is particularly suitable for real-time applications like traffic sign recognition and face mask detection. By applying transfer learning, YOLOv8 can be fine-tuned on specific datasets for custom object detection tasks.

•⁠  ⁠*FastRCNN - Neural Networks*
  - FastRCNN is a deep learning model designed for efficient object detection tasks. It integrates both region proposal generation and classification into a unified framework. Transfer learning with FastRCNN allows for quick adaptation to new domains, enhancing detection accuracy and reducing training time.
