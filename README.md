# KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System
The project aims to develop an Amharic NER (Named Entity Recognition) system to extract vital business entities (Product, Price, Location) from diverse Telegram channel content. This extracted, structured data will feed into EthioMart's centralized database, facilitating a seamless e-commerce experience for customers and vendors.
# EthioMart Amharic Named Entity Recognition System

This project aims to develop an Amharic Named Entity Recognition (NER) system to extract key business entities (Product, Price, Location, Material, Delivery Fee, Contact Info) from Telegram e-commerce channels in Ethiopia. This will centralize data for EthioMart's platform.

## Project Structure

KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── documents/
│   │   ├── telegram_data.csv          # Output from scraper
│   │   ├── telegram_messages.jsonl    # Converted from CSV for consistency
│   │   └── channels_to_crawl.xlsx
│   ├── processed/
│   │   ├── processed_messages.jsonl   # After OCR, normalization, and cleaning
│   │   └── ocr_output_logs/
│   ├── labeled/
│   │   ├── train.conll
│   │   ├── dev.conll
│   │   └── test.conll
│   └── external/
│       ├── amharic_news_ner_dataset/
│       └── shageronlinestore_sample/
├── notebooks/
│   ├── 01_data_ingestion_preprocessing.ipynb
│   ├── 02_data_labeling_guidelines.ipynb # You'd fill this with your labeling process/rules
│   ├── 03_model_finetuning.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_model_interpretability.ipynb
│   └── 06_vendor_scorecard_analytics.ipynb
├── scripts/
│   ├── telegram_scraper.py             # Completed as per last interaction
│   ├── data_preprocessing.py           # Handles CSV to JSONL, OCR, text cleaning
│   ├── data_splitting.py               # To split your labeled CoNLL data
│   ├── ner_model_training.py           # Main script for fine-tuning
│   ├── ner_evaluation.py
│   ├── vendor_analytics_engine.py
│   └── utils.py
├── models/
├── reports/
├── config/
│   ├── .env                           # IMPORTANT: Add to .gitignore!
│   └── project_config.ini
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE

### Key Directories Explained:

* **`data/`**: Contains all data related to the project.
    * `raw/`: Stores initial, unmodified data (e.g., scraped Telegram data, raw images/documents).
    * `processed/`: Holds cleaned, normalized, and preprocessed data, ready for modeling.
    * `labeled/`: Contains manually or semi-automatically labeled datasets in CoNLL format.
    * `external/`: For external datasets used for pre-training or comparative analysis.
* **`notebooks/`**: Jupyter notebooks for exploratory data analysis, experimentation, and detailed step-by-step processes.
* **`scripts/`**: Reusable Python scripts for automated tasks, data processing pipelines, model training, and evaluation.
* **`models/`**: Saved trained models, checkpoints, and model artifacts.
* **`reports/`**: Generated reports, figures, and other analysis outputs.
* **`config/`**: Configuration files for the project, including environment variables and project settings.
* **`.gitignore`**: Specifies files and directories that Git should ignore (e.g., `venv/`, `.env`, `__pycache__`).
* **`requirements.txt`**: Lists all Python dependencies required for the project.
* **`README.md`**: This file, providing an overview of the project.
* **`LICENSE`**: The license under which the project is distributed.


## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/benkenlin/KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System.git](https://github.com/benkenlin/KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System.git)
    cd KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Tesseract-OCR Engine:**
    Tesseract is a separate software. Follow the instructions for your OS:
    * **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-amh`
    * **macOS (Homebrew):** `brew install tesseract` (and ensure `amh.traineddata` is present in `tessdata`)
    * **Windows:** Download and run the installer from Tesseract-OCR GitHub Wiki, ensuring Amharic language pack and "Add to PATH" are selected.

5.  **Configure Telegram API Credentials:**
    * Go to [https://my.telegram.org/auth](https://my.telegram.org/auth) to get your `api_id` and `api_hash`.
    * Create a file `config/.env` in the project root with your credentials:
        ```
        TG_API_ID=YOUR_API_ID
        TG_API_HASH=YOUR_API_HASH
        TG_PHONE_NUMBER=+251XXXXXXXXX
        ```
    * **IMPORTANT:** Ensure `config/.env` is listed in your `.gitignore` file.

## Project Tasks

This project is divided into the following objectives:

* **Task 1: Data Ingestion and Preprocessing:** Collects and cleans Amharic data from Telegram.
* **Task 2: Label a Subset of Dataset in CoNLL Format:** Manual annotation for NER.
* **Task 3: Fine Tune NER Model:** Train a transformer model on labeled data.
* **Task 4: Model Comparison & Selection:** Evaluate and choose the best model.
* **Task 5: Model Interpretability:** Understand model decisions.
* **Task 6: FinTech Vendor Scorecard:** Apply NER output for business insights.

## How to Run

### Task 1: Data Ingestion & Preprocessing

1.  **Prepare Channels List:** Ensure `data/raw/channels_to_crawl.xlsx` exists and contains a 'Channel Username' column with the Telegram channel usernames (e.g., `@Shageronlinestore`).
2.  **Run Telegram Scraper:**
    ```bash
    python scripts/telegram_scraper.py
    ```
    This will generate `data/raw/telegram_data.csv` and download images/documents to `data/raw/images/` and `data/raw/documents/`.
3.  **Convert CSV to JSONL:**
    ```bash
    python scripts/data_preprocessing.py # This script will handle the conversion and OCR
    ```
    This will create `data/raw/telegram_messages.jsonl` and `data/processed/processed_messages.jsonl` (after OCR and cleaning).

### Task 2: Data Labeling

* Manually label a subset of messages from `data/raw/telegram_messages.jsonl` in CoNLL format, identifying `B-Product`, `I-Product`, `B-LOC`, `I-LOC`, `B-PRICE`, `I-PRICE`, and `O`.
* Save your labeled data as `data/labeled/raw_labeled_data.conll` (or similar).
* Use `scripts/data_splitting.py` to create `train.conll`, `dev.conll`, `test.conll` from your labeled data.
    ```bash
    python scripts/data_splitting.py
    ```

### Task 3: Fine-Tuning NER Model

* Open `notebooks/03_model_finetuning.ipynb` (or `scripts/ner_model_training.py`) and follow the steps to load labeled data, tokenize, set up Trainer, and fine-tune an XLM-RoBERTa (or other) model.

### (Continue for other tasks with brief instructions)

## Author

* **[Your Name]** - benkenlin

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Usage

(Instructions on how to run scripts)

## Contact

By Kenesa B.
getkennyo@gmail.com
