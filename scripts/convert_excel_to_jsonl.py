import pandas as pd
import json
import os

# Define file paths based on your project structure
# Define file paths based on your project structure
# Since you are running from the project root (KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System),
# the 'data/' folder should be directly inside it.
RAW_DATA_DIR = 'data/raw/'
EXCEL_FILE = os.path.join(RAW_DATA_DIR, 'telegram_data.xlsx')
JSONL_OUTPUT_FILE = os.path.join(RAW_DATA_DIR, 'telegram_messages.jsonl')

print(f"Attempting to convert: {EXCEL_FILE}")
# --- Add these lines to debug ---
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for file at: {os.path.abspath(EXCEL_FILE)}")
print(f"Does the 'data' folder exist?: {os.path.exists('data/')}")
print(f"Does the 'data/raw' folder exist?: {os.path.exists('data/raw/')}")
print(f"Does the file '{EXCEL_FILE}' exist?: {os.path.exists(EXCEL_FILE)}")
# --- End of debug lines ---


try:
    # 1. Load the Excel file into a pandas DataFrame
    # If your Excel has multiple sheets, you might need to specify sheet_name='Sheet1'
    df = pd.read_excel(EXCEL_FILE)
    print(f"Successfully loaded {EXCEL_FILE}. Shape: {df.shape}")
    print("Columns available:", df.columns.tolist())

    # 2. Inspect the DataFrame (Optional, but highly recommended)
    # This helps you understand the column names and data types
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

    # 3. Convert DataFrame to JSONL format
    # The 'records' orientation outputs a list of dictionaries (one dict per row)
    # The 'lines=True' argument makes it write one JSON object per line (JSONL)
    # ensure_ascii=False ensures Amharic characters are correctly encoded
    df.to_json(JSONL_OUTPUT_FILE, orient='records', lines=True, force_ascii=False)

    print(f"\nSuccessfully converted '{EXCEL_FILE}' to '{JSONL_OUTPUT_FILE}'")

    # Optional: Verify the first few lines of the JSONL file
    print(f"\nFirst 3 lines of '{JSONL_OUTPUT_FILE}':")
    with open(JSONL_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.strip())

except FileNotFoundError:
    print(f"Error: The file '{EXCEL_FILE}' was not found. Please ensure it's in the '{RAW_DATA_DIR}' directory.")
except Exception as e:
    print(f"An error occurred during conversion: {e}")