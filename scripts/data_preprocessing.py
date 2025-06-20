import pandas as pd
import json
import os
import re
from PIL import Image
import pytesseract
import logging
from langdetect import detect, DetectorFactory # NEW: Import langdetect

# Ensure reproducibility of language detection (optional but good practice)
DetectorFactory.seed = 0 # NEW: Seed for langdetect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
IMAGES_DIR = os.path.join(DATA_RAW_DIR, 'images')
DOCUMENTS_DIR = os.path.join(DATA_RAW_DIR, 'documents') # Assuming text can be extracted from docs too

CSV_INPUT_FILE = os.path.join(DATA_RAW_DIR, 'telegram_data.csv')
JSONL_OUTPUT_RAW_FILE = os.path.join(DATA_RAW_DIR, 'telegram_messages.jsonl')
JSONL_OUTPUT_PROCESSED_FILE = os.path.join(DATA_PROCESSED_DIR, 'processed_messages.jsonl')

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True) # Ensure processed data dir exists

# Path to your Tesseract executable (change if necessary)
# For Windows, uncomment and set the path if Tesseract is not in your system PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- Amharic Text Normalization Function ---
def normalize_amharic_text(text):
    if text is None:
        return ""
    text = str(text) # Ensure it's a string

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Basic cleaning (remove some non-Amharic non-punctuation characters)
    # This regex keeps Amharic characters (\u1200-\u137F), common punctuation, numbers, and basic Latin (for mixed content)
    # Adjust this regex as needed for specific noise in your data
    text = re.sub(r'[^\u1200-\u137F\s\w\d\.\,\?\!\:\;\(\)\@\#\%\&\*\-\+\=\/\<\>\'\"\`\~]', '', text)

    # Unicode normalization (NFC is generally good for Amharic)
    # This line below does a simple re-encoding to handle some common unicode issues.
    # For more robust Amharic normalization, you might need a dedicated library.
    text = text.strip().encode('utf-8').decode('utf-8') 

    return text

# --- NEW: Language Detection Function ---
def detect_language(text):
    """Detects the language of the given text."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        # Minimum text length for detection might be an issue for very short strings
        return detect(text)
    except Exception: # Catch any exceptions from langdetect (e.g., text too short or not enough unique words)
        return "unknown"


# --- OCR Function ---
def perform_ocr(image_path):
    """Performs OCR on an image file using Tesseract-OCR with Amharic language."""
    if not os.path.exists(image_path):
        logging.warning(f"Image not found for OCR: {image_path}")
        return ""
    try:
        img = Image.open(image_path)
        # Assuming 'amh' language data is installed for Tesseract
        text = pytesseract.image_to_string(img, lang='amh')
        logging.info(f"OCR successful for {os.path.basename(image_path)}")
        return text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. Please install it.")
        return ""
    except Exception as e:
        logging.error(f"Error during OCR for {image_path}: {e}")
        return ""

# --- Main Preprocessing Logic ---
def preprocess_telegram_data():
    logging.info(f"Starting data preprocessing.")

    # 1. Convert CSV to JSONL (telegram_data.csv -> telegram_messages.jsonl)
    logging.info(f"Converting '{CSV_INPUT_FILE}' to JSONL format...")
    try:
        df = pd.read_csv(CSV_INPUT_FILE)
        # Fill NaN in 'Message' with empty string to avoid issues
        df['Message'] = df['Message'].fillna('')
        df['Media Path'] = df['Media Path'].fillna('') # Handle NaN in media path

        # Convert DataFrame to JSONL format for raw messages
        df.to_json(JSONL_OUTPUT_RAW_FILE, orient='records', lines=True, force_ascii=False)
        logging.info(f"Successfully converted CSV to '{JSONL_OUTPUT_RAW_FILE}'.")
    except FileNotFoundError:
        logging.error(f"Error: Input CSV file not found at '{CSV_INPUT_FILE}'. Run scraper first.")
        return
    except Exception as e:
        logging.error(f"Error converting CSV to JSONL: {e}")
        return

    # 2. Process each message for OCR and Text Normalization
    processed_messages = []
    with open(JSONL_OUTPUT_RAW_FILE, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in):
            try:
                message_data = json.loads(line)

                # Initialize cleaned_text with original message text
                original_message_text = message_data.get('Message', '')
                cleaned_text = normalize_amharic_text(original_message_text)
                ocr_text = ""

                # Perform OCR if there's an image associated
                if message_data.get('Media Type') == 'photo' and message_data.get('Media Path'):
                    # The Media Path needs to be correctly resolved.
                    # Assuming `message_data['Media Path']` is something like `images/filename.jpg`
                    # or `photos/filename.jpg` relative to `data/raw/`.
                    # Adjust the path reconstruction based on how your scraper saves it.
                    
                    # Example: If scraper saves `images/filename.jpg` directly relative to PROJECT_ROOT
                    # full_image_path = os.path.join(PROJECT_ROOT, message_data['Media Path'])
                    
                    # Example: If scraper saves `photos/filename.jpg` into CSV, and actual files are in `data/raw/images/`
                    # This implies a mismatch or a renaming step in scraper's output
                    # Let's assume the scraper is saving into `data/raw/images/` and the CSV path is relative to `data/raw/`
                    media_relative_path = message_data['Media Path'] # e.g., 'images/channel_id.jpg'
                    full_image_path = os.path.join(DATA_RAW_DIR, media_relative_path)

                    ocr_text = perform_ocr(full_image_path)
                    ocr_text = normalize_amharic_text(ocr_text)
                    
                    if ocr_text:
                        cleaned_text = f"{cleaned_text} {ocr_text}".strip() # Combine text and OCR

                # NEW: Detect language of the final combined and cleaned text
                detected_lang = detect_language(cleaned_text)

                processed_message = {
                    "id": message_data.get('ID'),
                    "channel_title": message_data.get('Channel Title'),
                    "channel_username": message_data.get('Channel Username'),
                    "date": message_data.get('Date'),
                    "original_text": original_message_text, # Keep original for reference
                    "ocr_text": ocr_text, # Keep OCR text separate for reference
                    "text": cleaned_text, # The combined and cleaned text for NER
                    "media_path": message_data.get('Media Path'), # Original media path from CSV
                    "media_type": message_data.get('Media Type'),
                    "views": message_data.get('Views'),
                    "language": detected_lang # NEW: Add detected language
                }
                processed_messages.append(processed_message)

            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error on line {line_num+1}: {e} - Content: {line.strip()}")
            except Exception as e:
                logging.error(f"Error processing message on line {line_num+1}: {e}")

    # Optional: Filter for Amharic messages only here if desired
    # If you only want to process messages that are primarily Amharic for your NER
    # uncomment the following lines:
    # original_count = len(processed_messages)
    # processed_messages = [msg for msg in processed_messages if msg['language'] == 'am']
    # logging.info(f"Filtered down to {len(processed_messages)} Amharic messages from {original_count} total.")


    # 3. Save the fully processed data to processed/processed_messages.jsonl
    logging.info(f"Saving processed data to '{JSONL_OUTPUT_PROCESSED_FILE}'...")
    with open(JSONL_OUTPUT_PROCESSED_FILE, 'w', encoding='utf-8') as f_out:
        for msg in processed_messages:
            f_out.write(json.dumps(msg, ensure_ascii=False) + '\n')
    logging.info(f"Data preprocessing complete. {len(processed_messages)} messages processed.")

if __name__ == '__main__':
    preprocess_telegram_data()