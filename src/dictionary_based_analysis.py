import os
import pandas as pd
from collections import Counter
import math
import warnings

def get_hawkish_score(dictionary_path, text_files_dir) -> pd.DataFrame:
    # Read the words into a list
    with open(dictionary_path, 'r') as file:
        words_list = [line.strip() for line in file.readlines()]

    # Initialize word counts dictionary for hawkish words
    word_counts = {word: [] for word in words_list}

    # Initialize total word count list for each document
    total_word_count = []

    # Get list of .txt files
    txt_files = [f for f in os.listdir(text_files_dir) if f.endswith('.txt')]

    # Count occurrences of each hawkish word in each file and total words
    for txt_file in txt_files:
        with open(os.path.join(text_files_dir, txt_file), 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Convert to lowercase

        word_counter = Counter(text.split())
        
        # Append total word count for this document
        total_word_count.append(len(text.split()))

        # Append hawkish word counts to the dictionary
        for word in words_list:
            word_counts[word].append(word_counter.get(word.lower(), 0))

    # Convert word counts to DataFrame
    count_matrix_df = pd.DataFrame(word_counts, index=txt_files)

    # Create DataFrame for total word counts
    stats_df = pd.DataFrame({'Total_Words': total_word_count}, index=txt_files)

    # Save word count matrix and total word counts to CSV
    count_matrix_df.to_csv('word_count_matrix.csv')
    stats_df.to_csv('file_word_count.csv')

    # Display the stats DataFrame
    print(stats_df.head())


    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Calculate TF-IDF
    df_tf_idf = count_matrix_df.copy()
    df_i = [0] * count_matrix_df.shape[1]

    for i in range(df_tf_idf.shape[0]):  # Iterate over documents
        for j in range(df_tf_idf.shape[1]):  # Iterate over words
            if df_tf_idf.iloc[i, j]:
                df_tf_idf.iloc[i, j] = (1 + math.log(df_tf_idf.iloc[i, j])) / (1 + math.log(stats_df['Total_Words'][i]))
                df_i[j] += 1

    df_i = [math.log(df_tf_idf.shape[0] / i) if i else 0 for i in df_i]
    df_tf_idf = df_tf_idf.mul(df_i, axis=1)

    print(df_tf_idf.head(5))
    print(df_tf_idf.shape)

    ### Weighting
    # Multiply word counts by weights (TF-IDF)
    weighted_counts = df_tf_idf * count_matrix_df

    # Sum weighted counts for each document
    weighted_sum = weighted_counts.sum(axis=1)

    # Create a DataFrame for the results
    weighted_sum_df = pd.DataFrame(weighted_sum, columns=['Weighted_Hawkish_Sum'])

    # # Display the result
    # print(weighted_sum_df.head())
    # print(weighted_sum_df.tail())

    return weighted_sum_df

if __name__ == "__main__":
    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/fomc_press_conf/texts') 
    print(df.head(10))
    print(df.tail(10))

    df = get_hawkish_score('data/processed/hawkish_gpt_dict.txt','data/raw/individual/meeting_minutes') 
    print(df.head(10))
    print(df.tail(10))