import pandas as pd
import json
import os
import numpy as np
from config import *

def load_csv_data(csv_path):
    return pd.read_csv(csv_path)

def calculate_sentiment_scores(sentences):
    positive_score = 0
    negative_score = 0

    for sentence in sentences:
        sentiment = sentence.get('sentiment', {})
        score = sentiment.get('score', 0)
        magnitude = sentiment.get('magnitude', 0)

        if magnitude > 0 and score != 0:
            if score > 0:
                positive_score += score
            elif score < 0:
                negative_score += score
    
    return positive_score, negative_score

def read_sentiment_json(pet_id, json_folder_path='./data/train_sentiment/'):
    json_file = os.path.join(json_folder_path, f'{pet_id}.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Sentiment file not found. Please provide a valid petid: {pet_id}")
    with open(json_file, 'r', encoding='utf-8') as file:
        sentiment_data = json.load(file)
    return sentiment_data

def add_sentiment_scores_to_dataframe(df, json_folder_path='./data/train_sentiment/'):
    copied = df.copy()
    positive_scores = []
    negative_scores = []
    for pet_id in df['PetID']:
        try:
            sentiment_data = read_sentiment_json(pet_id, json_folder_path)
            sentences = sentiment_data.get('sentences', [])
            positive_score, negative_score = calculate_sentiment_scores(sentences) 
        except FileNotFoundError as e:
            positive_score, negative_score = np.nan, np.nan
            print(e)
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
    copied.loc[:, 'PositiveSentimentScore'] = positive_scores
    copied.loc[:, 'NegativeSentimentScore'] = negative_scores
    target_column = copied.pop(target[0])
    copied[target[0]] = target_column
    return copied

if __name__ == "__main__":
    df = load_csv_data(TRAIN_PATH)
    df = add_sentiment_scores_to_dataframe(df, SENTIMENT_FOLDER_PATH)
    df = df.drop(id_features+textual_features, axis=1)
    df = df.dropna()
    df.to_csv(SENTIMENT_INCLUDED_TRAIN_PATH, index=False)
    print(f"Processed data saved to {SENTIMENT_INCLUDED_TRAIN_PATH}")