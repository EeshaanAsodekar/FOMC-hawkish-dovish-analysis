import pandas as pd
import os

# Clean the raw FOMC data from CSV files
def clean_fomc_data(filepath, doc_type):
    """
    Cleans the raw FOMC data from the provided CSV file by:
    - Renaming 'Unnamed: 0' to 'Date'.
    - Checking for NaN values and printing their count.
    - Dropping rows where 'Date' or the content is missing.
    - Removing duplicate rows.
    
    Args:
    filepath (str): Path to the raw CSV file to be cleaned.
    doc_type (str): The type of document (e.g., "Minutes", "Statements").
    
    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # Load the raw data
    data = pd.read_csv(filepath)
    
    # Rename 'Unnamed: 0' to 'Date'
    data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    
    # Print the count of NaN values in each column
    print(f"NaN counts for {doc_type}:")
    print(data.isna().sum())
    
    # Drop rows where 'Date' or the content is missing
    if doc_type == "Minutes":
        data.dropna(subset=['Date', 'Federal_Reserve_Mins'], inplace=True)
    elif doc_type == "Statements":
        data.dropna(subset=['Date', 'FOMC_Statements'], inplace=True)
    
    # Drop duplicate rows
    initial_count = data.shape[0]
    data.drop_duplicates(inplace=True)
    print(f"Dropped {initial_count - data.shape[0]} duplicate rows from {doc_type}.")
    
    return data

# Save cleaned data back to a CSV file
def save_cleaned_data(input_file, output_file, doc_type):
    """
    Cleans the input data and saves the cleaned version to the specified output file.
    
    Args:
    input_file (str): Path to the raw input CSV file.
    output_file (str): Path to save the cleaned CSV file.
    doc_type (str): The type of document (e.g., "Minutes", "Statements").
    """
    cleaned_data = clean_fomc_data(input_file, doc_type)
    cleaned_data.to_csv(output_file, index=False)  # Save cleaned data without index
    print(f"Cleaned data saved to {output_file}")

# Process all raw FOMC Meeting Minutes and Statements
def process_all_documents():
    """
    Processes both FOMC Meeting Minutes and FOMC Statements by cleaning the raw CSV files
    and saving the cleaned versions into the processed data directory.
    """
    files_to_process = {
        'FOMC_meeting_minutes.csv': ('cleaned_meeting_minutes.csv', 'Minutes'),
        'FOMC_statements.csv': ('cleaned_statements.csv', 'Statements')
    }

    # Loop through each file in the mapping of raw to cleaned file paths
    for raw_file, (cleaned_file, doc_type) in files_to_process.items():
        input_path = os.path.join('data/raw', raw_file)
        output_path = os.path.join('data/processed', cleaned_file)

        # Ensure the input file exists before attempting to process
        if os.path.exists(input_path):
            save_cleaned_data(input_path, output_path, doc_type)
        else:
            print(f"File {input_path} does not exist, skipping...")
# Function to wrap content with a given number of words per line
def wrap_text(text, words_per_line=10):
    """
    Wraps the text such that each line contains a specified number of words.
    
    Args:
    text (str): The content to be wrapped.
    words_per_line (int): The number of words per line. Default is 10.

    Returns:
    str: The wrapped text with the specified number of words per line.
    """
    words = text.split()  # Split the content into words
    wrapped_lines = []

    # Loop through the words and create lines of words_per_line
    for i in range(0, len(words), words_per_line):
        wrapped_lines.append(" ".join(words[i:i + words_per_line]))

    return "\n".join(wrapped_lines)  # Join the lines with newlines


# Save each row of data as an individual text file
def save_individual_files(input_file, output_dir, doc_type):
    """
    Saves each row of the input CSV as an individual text file within the specified output directory.
    Each text file will be named using the date and the document type (e.g., '2024-05-01_Minutes.txt').
    
    Args:
    input_file (str): Path to the cleaned CSV file whose rows will be saved as individual files.
    output_dir (str): Directory where the individual text files will be saved.
    doc_type (str): The type of document (e.g., "Minutes", "Statements").
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the cleaned CSV file
    data = pd.read_csv(input_file)

    # Iterate through each row in the DataFrame
    for idx, row in data.iterrows():
        # Construct a unique filename using the 'Date' and doc_type
        date_str = row['Date']
        file_name = f"{date_str}_{doc_type}.txt"
        output_path = os.path.join(output_dir, file_name)

        # Get the relevant content (either minutes or statements)
        content = row['Federal_Reserve_Mins'] if doc_type == "Minutes" else row['FOMC_Statements']
        
        # Wrap the content to ensure 10 words per line
        wrapped_content = wrap_text(content, words_per_line=10)

        # Write the wrapped content to a text file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(wrapped_content)

    print(f"Saved individual files in {output_dir}")


# Rest of the module remains unchanged
def create_individual_files_for_minutes_and_statements():
    """
    Creates individual text files for each row of the cleaned FOMC Meeting Minutes and Statements.
    Saves these files into the `data/raw/FOMC` directory, under separate folders for 
    meeting minutes and statements.
    """
    minutes_input = 'data/processed/cleaned_meeting_minutes.csv'
    statements_input = 'data/processed/cleaned_statements.csv'

    # Create individual text files for meeting minutes and statements
    save_individual_files(minutes_input, 'data/raw/FOMC/meeting_minutes', 'Minutes')
    save_individual_files(statements_input, 'data/raw/FOMC/statements', 'Statements')

if __name__ == "__main__":
    # Process and clean both raw data files
    process_all_documents()
    
    # Create individual files for easier reading and analysis
    create_individual_files_for_minutes_and_statements()
