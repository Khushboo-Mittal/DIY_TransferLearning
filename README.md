# Transfer Learning 

This project leverages transfer learning techniques for two key tasks: **mask detection in public spaces** and **sentiment analysis of Twitter data**. It employs state-of-the-art models such as BERT for NLP and YOLOv8 for Computer Vision to make real-time predictions and analyze customer sentiment, aiding both public safety and digital marketing efforts.

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


## Program Flow

