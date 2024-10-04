import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load pre-trained FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertModel.from_pretrained('yiyanghkust/finbert-tone')

# Define Hawkish and Dovish sentences
HAWKISH_SENTENCES = [
    "Interest rates will rise.",
    "The economy is overheating.",
    "We will tighten monetary policy.",
    "Inflation is high.",
    "We are concerned about rising prices."
]

DOVISH_SENTENCES = [
    "Interest rates will fall.",
    "The economy is slowing down.",
    "We will implement quantitative easing.",
    "Inflation is low.",
    "We are concerned about slow growth."
]

# Function to generate embeddings from text using FinBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling to get single embedding

# Function to calculate similarity between text and hawkish/dovish sentences
def calculate_similarity(text, hawkish_sentences, dovish_sentences):
    text_embedding = get_embedding(text)

    hawkish_scores = [cosine_similarity(text_embedding, get_embedding(sentence)) for sentence in hawkish_sentences]
    dovish_scores = [cosine_similarity(text_embedding, get_embedding(sentence)) for sentence in dovish_sentences]

    avg_hawkish = np.mean(hawkish_scores)
    avg_dovish = np.mean(dovish_scores)

    return avg_hawkish, avg_dovish

# Main function to process CSVs and calculate factor similarity scores
def process_fomc_documents():
    # Load the cleaned data for Meeting Minutes and Statements
    minutes_df = pd.read_csv('data/processed/cleaned_meeting_minutes.csv')
    statements_df = pd.read_csv('data/processed/cleaned_statements.csv')

    # Add columns for storing Hawkish and Dovish scores
    minutes_df['Hawkish_Score'] = 0.0
    minutes_df['Dovish_Score'] = 0.0
    statements_df['Hawkish_Score'] = 0.0
    statements_df['Dovish_Score'] = 0.0

    # Calculate similarity scores for each row in the Meeting Minutes
    print("Processing FOMC Meeting Minutes...")
    for idx, row in minutes_df.iterrows():
        hawkish, dovish = calculate_similarity(row['Federal_Reserve_Mins'], HAWKISH_SENTENCES, DOVISH_SENTENCES)
        minutes_df.at[idx, 'Hawkish_Score'] = hawkish
        minutes_df.at[idx, 'Dovish_Score'] = dovish
        if idx % 10 == 0:  # Progress log
            print(f"Processed {idx + 1} meeting minutes.")

    # Calculate similarity scores for each row in the FOMC Statements
    print("Processing FOMC Statements...")
    for idx, row in statements_df.iterrows():
        hawkish, dovish = calculate_similarity(row['FOMC_Statements'], HAWKISH_SENTENCES, DOVISH_SENTENCES)
        statements_df.at[idx, 'Hawkish_Score'] = hawkish
        statements_df.at[idx, 'Dovish_Score'] = dovish
        if idx % 10 == 0:  # Progress log
            print(f"Processed {idx + 1} statements.")

    # Save the results back to CSV
    minutes_df.to_csv('data/processed/hawkish_dovish_meeting_minutes.csv', index=False)
    statements_df.to_csv('data/processed/hawkish_dovish_statements.csv', index=False)

    print("Factor similarity analysis complete. Results saved.")

if __name__ == "__main__":
    process_fomc_documents()
