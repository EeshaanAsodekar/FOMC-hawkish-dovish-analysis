# src/data_processing.py
import pandas as pd
import os

# Clean the raw FOMC data from CSV files
def clean_fomc_data(filepath):
    """
    Cleans the raw FOMC data from the provided CSV file by:
    - Removing duplicate rows (if any).
    - Filling missing values with empty strings.
    
    Args:
    filepath (str): Path to the raw CSV file to be cleaned.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with duplicates removed and missing values handled.
    """
    data = pd.read_csv(filepath)
    data.drop_duplicates(inplace=True)  # Remove duplicate rows to avoid redundancy
    data.fillna('', inplace=True)       # Fill missing values with an empty string for uniformity
    return data

# Save cleaned data back to a CSV file
def save_cleaned_data(input_file, output_file):
    """
    Cleans the input data and saves the cleaned version to the specified output file.
    
    Args:
    input_file (str): Path to the raw input CSV file.
    output_file (str): Path to save the cleaned CSV file.
    """
    cleaned_data = clean_fomc_data(input_file)
    cleaned_data.to_csv(output_file, index=False)  # Save the cleaned data without row indices
    print(f"Cleaned data saved to {output_file}")

# Process all raw FOMC Meeting Minutes and Statements
def process_all_documents():
    """
    Processes both FOMC Meeting Minutes and FOMC Statements by cleaning the raw CSV files
    and saving the cleaned versions into the processed data directory.
    
    This function checks the existence of the input files before processing and skips
    any file that does not exist.
    """
    files_to_process = {
        'FOMC_meeting_minutes.csv': 'cleaned_meeting_minutes.csv',
        'FOMC_statements.csv': 'cleaned_statements.csv'
    }

    # Loop through each file in the mapping of raw to cleaned file paths
    for raw_file, cleaned_file in files_to_process.items():
        input_path = os.path.join('data/raw', raw_file)
        output_path = os.path.join('data/processed', cleaned_file)

        # Ensure the input file exists before attempting to process
        if os.path.exists(input_path):
            save_cleaned_data(input_path, output_path)
        else:
            print(f"File {input_path} does not exist, skipping...")

# Save each row of data as an individual text file
def save_individual_files(input_file, output_dir):
    """
    Saves each row of the input CSV as an individual text file within the specified output directory.
    Each text file will contain the data from one row, formatted as key-value pairs.

    Args:
    input_file (str): Path to the cleaned CSV file whose rows will be saved as individual files.
    output_dir (str): Directory where the individual text files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the cleaned CSV file
    data = pd.read_csv(input_file)

    # Iterate through each row in the DataFrame
    for idx, row in data.iterrows():
        # Construct a unique filename using the row index
        file_name = f"document_{idx}.txt"
        output_path = os.path.join(output_dir, file_name)

        # Convert the row's data to a string with each column value on a new line
        row_content = "\n".join([f"{col}: {row[col]}" for col in data.columns])

        # Write the row's data to a text file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(row_content)

    print(f"Saved individual files in {output_dir}")

# Create individual files for both FOMC Meeting Minutes and Statements
def create_individual_files_for_minutes_and_statements():
    """
    Creates individual text files for each row of the cleaned FOMC Meeting Minutes and Statements.
    Saves these files into the `data/raw/individual` directory, under separate folders for 
    meeting minutes and statements.
    """
    minutes_input = 'data/processed/cleaned_meeting_minutes.csv'
    statements_input = 'data/processed/cleaned_statements.csv'

    # Create individual text files for meeting minutes and statements
    save_individual_files(minutes_input, 'data/raw/individual/meeting_minutes')
    save_individual_files(statements_input, 'data/raw/individual/statements')


if __name__ == "__main__":
    # Process and clean both raw data files
    process_all_documents()
    
    # Create individual files for easier reading and analysis
    create_individual_files_for_minutes_and_statements()
