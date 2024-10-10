import os
import pandas as pd
from collections import Counter
import math
import warnings

def get_hawkish_score(dictionary_path, text_files_dir) -> pd.DataFrame:
    """
    Calculate hawkish word scores for text documents using a provided word dictionary.
    
    Args:
    dictionary_path (str): Path to the dictionary file containing hawkish words.
    text_files_dir (str): Directory containing text files to analyze.

    Returns:
    pd.DataFrame: DataFrame containing the weighted hawkish word score for each document.
    """
    
    # Load the hawkish words from the dictionary file into a list
    with open(dictionary_path, 'r') as file:
        words_list = [line.strip() for line in file.readlines()]

    # Initialize a dictionary to store hawkish word counts for each document
    word_counts = {word: [] for word in words_list}

    # List to store total word count for each document
    total_word_count = []

    # Get all .txt files in the specified directory
    txt_files = [f for f in os.listdir(text_files_dir) if f.endswith('.txt')]

    # Count occurrences of each hawkish word in each document and compute total word count
    for txt_file in txt_files:
        with open(os.path.join(text_files_dir, txt_file), 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Convert text to lowercase

        word_counter = Counter(text.split())
        
        # Append total word count for the current document
        total_word_count.append(len(text.split()))

        # Count occurrences of each hawkish word and store in word_counts
        for word in words_list:
            word_counts[word].append(word_counter.get(word.lower(), 0))

    # Create a DataFrame with hawkish word counts for each document
    count_matrix_df = pd.DataFrame(word_counts, index=txt_files)

    # Create a DataFrame for total word counts in each document
    stats_df = pd.DataFrame({'Total_Words': total_word_count}, index=txt_files)

    # # Save word counts and total word counts to CSV files (optional)
    # count_matrix_df.to_csv('word_count_matrix.csv')
    # stats_df.to_csv('file_word_count.csv')

    # Display the total word counts DataFrame
    # print(stats_df.head())

    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Calculate TF-IDF for hawkish word counts
    df_tf_idf = count_matrix_df.copy()
    df_i = [0] * count_matrix_df.shape[1]

    for i in range(df_tf_idf.shape[0]):  # Iterate over documents
        for j in range(df_tf_idf.shape[1]):  # Iterate over words
            if df_tf_idf.iloc[i, j]:
                # Apply TF-IDF formula
                df_tf_idf.iloc[i, j] = (1 + math.log(df_tf_idf.iloc[i, j])) / (1 + math.log(stats_df['Total_Words'][i]))
                df_i[j] += 1

    # Calculate IDF for each word
    df_i = [math.log(df_tf_idf.shape[0] / i) if i else 0 for i in df_i]
    df_tf_idf = df_tf_idf.mul(df_i, axis=1)

    # Display the TF-IDF matrix for the first 5 documents
    # print(df_tf_idf.head(5))
    # print(df_tf_idf.shape)

    # Weighting: Multiply TF-IDF by word counts to calculate weighted hawkish word scores
    weighted_counts = df_tf_idf * count_matrix_df

    # Sum the weighted scores for each document
    weighted_sum = weighted_counts.sum(axis=1)

    # Create a DataFrame to store the final weighted hawkish score
    weighted_sum_df = pd.DataFrame(weighted_sum, columns=['Weighted_Hawkish_Sum'])

    # Return the DataFrame containing the weighted hawkish scores
    return weighted_sum_df


if __name__ == "__main__":
    print("Hawkish Scores for Fed Chair Press Conferences")
    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/fomc_press_conf/texts') 
    print(df.head(10))
    print(df.tail(10))
    df.to_csv('data/results/dict-hawkish-scored_Fed-chair-press-conf.csv')


    print("\n*****************************************\n")
    print("Hawkish Scores for FOMC Meeting Minutes")
    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/individual/meeting_minutes') 
    print(df.head(10))
    print(df.tail(10))
    df.to_csv('data/results/dict-hawkish-scored_FOMC-meeting-minutes.csv')
    print("\n*****************************************\n")


    print("\n*****************************************\n")
    print("Hawkish Scores for FOMC Statements")
    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/individual/statements') 
    print(df.head(10))
    print(df.tail(10))
    df.to_csv('data/results/dict-hawkish-scored_FOMC-statements.csv')
    print("\n*****************************************\n")


    print("\n*****************************************\n")
    print("Hawkish Scores for Fed Speeches")
    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/fed_speeches') 
    print(df.head(10))
    print(df.tail(10))
    df.to_csv('data/results/dict-hawkish-scored_Fed-speeches.csv')
    print("\n*****************************************\n")
