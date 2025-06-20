import pandas as pd
import json
import os
import re
from PIL import Image
import pytesseract
import logging

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
    text = text.strip().encode('utf-8').decode('utf-8') # Simple NFC-like conversion for common issues

    return text

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
                cleaned_text = normalize_amharic_text(message_data.get('Message', ''))
                ocr_text = ""

                # Perform OCR if there's an image associated
                if message_data.get('Media Type') == 'photo' and message_data.get('Media Path'):
                    # Reconstruct full path relative to PROJECT_ROOT
                    # The Media Path stored in CSV is already relative to DATA_RAW_DIR
                    # so, if it's `images/filename.jpg`, we need to join it from DATA_RAW_DIR.
                    # No, the scraper saves it as `photos/filename.jpg` directly, which is problematic.
                    # It should save `data/raw/images/filename.jpg`
                    # For now, let's assume `message_data['Media Path']` is relative to the `PROJECT_ROOT`
                    # if scraper saves like `images/filename.jpg`
                    
                    # Fix: The scraper's `media_path` is `photos/filename` relative to the *scraper's run directory*.
                    # To fix this, the scraper should save paths like `data/raw/images/filename`.
                    # For this `data_preprocessing.py`, let's assume `message_data['Media Path']` is correct and relative to project root or use `os.path.join(DATA_RAW_DIR, message_data['Media Path'])`
                    
                    # Correct path for OCR:
                    full_image_path = os.path.join(DATA_RAW_DIR, message_data['Media Path'])
                    # If scraper saved `photos/image.jpg` then it should be `data/raw/photos/image.jpg`
                    # But our structure uses `data/raw/images/`
                    # So, if scraper saves 'photos/xyz.jpg', and we want it in 'data/raw/images/xyz.jpg'
                    # we need to adjust the path here or in scraper.
                    # Let's assume the scraper already downloads to `data/raw/images/` correctly now based on updated scraper.
                    # The scraper uses `media_dir = 'photos'` currently. Let's make it `IMAGES_DIR`.
                    # For current scraper, it writes `photos/filename.jpg` into CSV, so adjust the path here:
                    
                    # Fix for scraper's old `media_dir` naming:
                    if os.path.basename(os.path.dirname(message_data['Media Path'])) == 'photos':
                         fixed_media_path = os.path.join(IMAGES_DIR, os.path.basename(message_data['Media Path']))
                    else: # If scraper already produces data/raw/images/ structure
                         fixed_media_path = os.path.join(PROJECT_ROOT, message_data['Media Path'])
                    
                    ocr_text = perform_ocr(fixed_media_path) # Use the fixed path
                    ocr_text = normalize_amharic_text(ocr_text)
                    if ocr_text:
                        cleaned_text = f"{cleaned_text} {ocr_text}".strip()

                # Document OCR (placeholder)
                # if message_data.get('Media Type') == 'document' and message_data.get('Media Path'):
                #     # This would require a library like PyPDF2 or python-docx to extract text
                #     # from document types like PDF or Word.
                #     # For now, skipping document OCR unless explicitly implemented.
                #     pass

                message_data['cleaned_text'] = cleaned_text
                message_data['ocr_text'] = ocr_text # Keep OCR text separate for debugging/analysis
                processed_messages.append(message_data)

            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error on line {line_num+1}: {e} - Content: {line.strip()}")
            except Exception as e:
                logging.error(f"Error processing message on line {line_num+1}: {e}")

    # 3. Save the fully processed data to processed/processed_messages.jsonl
    logging.info(f"Saving processed data to '{JSONL_OUTPUT_PROCESSED_FILE}'...")
    with open(JSONL_OUTPUT_PROCESSED_FILE, 'w', encoding='utf-8') as f_out:
        for msg in processed_messages:
            f_out.write(json.dumps(msg, ensure_ascii=False) + '\n')
    logging.info(f"Data preprocessing complete. {len(processed_messages)} messages processed.")

if __name__ == '__main__':
    preprocess_telegram_data()