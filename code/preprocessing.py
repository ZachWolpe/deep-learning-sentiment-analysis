"""
----------------------------------------------------------------------------------
Preprocessing modules

This module contains functions to preprocess the data.

Functions:
    clean_text(text): Clean the text data

Usage:
    Execute the preprocessing pipeline from the command line (run from root directory)/
    
    `python code/preprocessing.py`

    Arguments:
       [1] --input_file (str): './Womens Clothing E-Commerce Reviews.csv' (default): Path to the input file
       [2] --output_file (str): './kaggle/working/womenclothing/customer_reviews.csv' (default): Path to the output file

: 22.05.24
: zachcolinwolpe@medibio.com
----------------------------------------------------------------------------------
"""

import pandas as pd
import nltk
import logging
import os
import argparse
import string
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


# Function to clean the text
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
    # Remove special characters and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Define the function to map ratings to sentiment labels
def map_rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


if __name__ == "__main__":
    # default args
    input_file = './data/raw/Womens Clothing E-Commerce Reviews.csv'
    output_file = './data/processed/customer_reviews.csv'

    # Parse the input and output file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=False, help="Path to the input file", default=input_file)
    parser.add_argument("--output_file", type=str, required=False, help="Path to the output file", default=output_file)
    parser.add_argument("--log_level", type=str, required=False, help="Logging level", default='INFO')
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(level=args.log_level)
    logging.info(f"Running preprocessing pipeline: {args.input_file} -> {args.output_file}...")

    # Read the input file
    df = pd.read_csv(args.input_file)

    # Drop rows with null values
    df.dropna(inplace=True)

    # Prepare the data for sentiment analysis --------------------------->>
    # Apply the cleaning function to the 'Text' column
    df['Review Text'] = df['Review Text'].apply(clean_text)
    # df.to_csv('./kaggle/working/womenclothing/customer_reviews.csv', index=False)

    df['Rating'] = df['Rating'].astype(int)
    df['Rating'] = df['Rating'].astype(int)

    # Map the ratings to sentiment labels and add the column to the DataFrame
    df["Sentiment"] = df["Rating"].apply(map_rating_to_sentiment)

    # Select the "Sentiment", "Rating", and "Review Text" columns
    df = df[["Sentiment", "Rating", "Review Text"]]
    # Prepare the data for sentiment analysis --------------------------->>


    # Store clean data -------------------------------------------------->>
    # create interim directory
    output_file_path = '/'.join(args.output_file.split('/')[:-1])
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Save the cleaned data to the output file
    df.to_csv(args.output_file, index=False)
    # Store clean data -------------------------------------------------->>

    # Log the dataframe shape, columns, and head
    log_info = f"""
        Dataframe Metadata: ------------------------------------------------
        : dataframe shape:      {df.shape}
        : dataframe columns:    {df.columns}
        : dataframe['Ratings']:
            value_counts:       {df['Rating'].value_counts()}
            isnull:             {df['Rating'].isnull().sum()}
        : dataframe head:       {df.head(2)}
        : dataframe location:   {args.output_file}
        --------------------------------------------------------------------
    """
    logging.info(log_info)
    logging.info("Preprocessing pipeline completed.")
