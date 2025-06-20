import pandas as pd
import json
import os
from transformers import pipeline
import datetime
from collections import defaultdict
import numpy as np

# --- Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_messages.jsonl')
# Path to your best fine-tuned NER model (e.g., the final_model subfolder from training)
BEST_NER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xlm_roberta_ner', 'final_model') # Make sure this is correct
# Output for the vendor scorecard
VENDOR_SCORECARD_OUTPUT = os.path.join(PROJECT_ROOT, 'reports', 'vendor_scorecard_summary.csv')

# Ensure reports directory exists
os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)

# --- Load NER Model (Pipeline for easy inference) ---
ner_pipeline = None
try:
    # We must explicitly define the ID2LABEL and LABEL2ID if they are not stored with the model config
    # Ensure this matches the labels used during training
    FINAL_NER_LABELS = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']
    ID2LABEL_PIPELINE = {i: label for i, label in enumerate(FINAL_NER_LABELS)}
    
    ner_pipeline = pipeline(
        "ner", 
        model=BEST_NER_MODEL_PATH, 
        tokenizer=BEST_NER_MODEL_PATH, 
        aggregation_strategy="simple",
        # Pass ID2LABEL if the model's config doesn't have it or needs override
        id2label=ID2LABEL_PIPELINE 
    )
    print(f"Loaded NER pipeline from {BEST_NER_MODEL_PATH}")
except Exception as e:
    print(f"Error loading NER model from {BEST_NER_MODEL_PATH}: {e}")
    print("Please ensure the model is trained and saved correctly at this path.")
    print("Proceeding without NER for analysis. Some metrics will be missing.")


# --- Helper function to extract entities and parse prices ---
def extract_entities_and_prices(text):
    if not ner_pipeline or not text:
        return [], []

    entities = ner_pipeline(text)
    products = []
    prices = []
    locations = [] # Also capture locations for potential future use

    for ent in entities:
        # ent['entity_group'] is from aggregation_strategy="simple" which converts B-X, I-X to X
        if ent['entity_group'] == 'PRODUCT':
            products.append(ent['word'].strip())
        elif ent['entity_group'] == 'PRICE':
            price_str = ent['word'].strip()
            # Clean and parse price (e.g., "ዋጋ 1000 ብር", "1,500.00", "5000ብር")
            price_str = price_str.replace("ዋጋ", "").replace("ብር", "").replace("ETB", "").replace("በ", "").strip()
            price_str = price_str.replace(",", "") # Remove commas for numerical conversion
            try:
                # Use a regex to capture numerical parts only
                num_match = re.search(r'\d+(\.\d+)?', price_str)
                if num_match:
                    prices.append(float(num_match.group(0)))
            except ValueError:
                pass # Ignore if price can't be converted to number
        elif ent['entity_group'] == 'LOC':
            locations.append(ent['word'].strip())
            
    return products, prices, locations

# --- Main Analytics Engine ---
def analyze_vendors():
    print("Loading processed message data...")
    messages = []
    try:
        with open(PROCESSED_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                messages.append(json.loads(line))
        df = pd.DataFrame(messages)
        print(f"Loaded {len(df)} messages.")

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date for consistent time-based metrics
        df = df.sort_values(by='Date')

    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{PROCESSED_DATA_FILE}'. Please run data_preprocessing.py first.")
        return
    except Exception as e:
        print(f"Error loading or parsing processed data: {e}")
        return

    vendor_metrics = defaultdict(lambda: {
        'total_posts': 0,
        'first_post_date': None,
        'last_post_date': None,
        'total_views': 0,
        'max_views_post_text': None,
        'max_views_post_products': [], # Store products from max view post
        'max_views_post_prices': [],   # Store prices from max view post
        'all_prices': [],
        'products_listed': []
    })

    print("Analyzing messages and extracting entities (this may take time)...")
    # Iterate through messages to populate vendor_metrics
    for index, row in df.iterrows():
        channel_username = row['Channel Username']
        message_text = row['cleaned_text'] if 'cleaned_text' in row else row['Message']
        post_date = row['Date']
        views = row['Views'] if pd.notna(row['Views']) else 0

        vendor_data = vendor_metrics[channel_username]
        vendor_data['total_posts'] += 1
        vendor_data['total_views'] += views

        if vendor_data['first_post_date'] is None or post_date < vendor_data['first_post_date']:
            vendor_data['first_post_date'] = post_date
        if vendor_data['last_post_date'] is None or post_date > vendor_data['last_post_date']:
            vendor_data['last_post_date'] = post_date

        products, prices, _ = extract_entities_and_prices(message_text)
        vendor_data['products_listed'].extend(products)
        vendor_data['all_prices'].extend(prices)

        if views > vendor_data.get('max_views', -1):
            vendor_data['max_views'] = views
            vendor_data['max_views_post_text'] = message_text
            vendor_data['max_views_post_products'] = products # Store extracted products
            vendor_data['max_views_post_prices'] = prices     # Store extracted prices

    final_scorecard_data = []

    print("Calculating final metrics for each vendor...")
    for channel_username, data in vendor_metrics.items():
        # Activity & Consistency: Posting Frequency
        time_span_days = (data['last_post_date'] - data['first_post_date']).days if data['first_post_date'] and data['last_post_date'] else 0
        posting_frequency_per_week = (data['total_posts'] / time_span_days * 7) if time_span_days > 0 else data['total_posts']

        # Market Reach & Engagement: Average Views per Post
        avg_views_per_post = data['total_views'] / data['total_posts'] if data['total_posts'] > 0 else 0

        # Business Profile: Average Price Point
        avg_price_point = np.mean(data['all_prices']) if data['all_prices'] else 0

        # Top Performing Post Details
        top_post_product_str = ", ".join(set(data['max_views_post_products'])) if data['max_views_post_products'] else "N/A"
        top_post_price_str = f"{np.mean(data['max_views_post_prices']):.2f}" if data['max_views_post_prices'] else "N/A"

        # --- Create a Simple "Lending Score" ---
        # This is a placeholder; you'll need to define this based on real business logic
        # and potentially normalize metrics across all vendors first.
        # Example Score Components:
        # - Higher posting frequency -> more active
        # - Higher average views -> more popular/engaged
        # - Relevant product/price info being extracted (implicit through NER quality)
        
        lending_score = (avg_views_per_post * 0.01) + (posting_frequency_per_week * 0.5) # Example weights
        if avg_price_point > 0: # Factor in price, maybe lower if prices are too high or too low, or just presence
            lending_score += (avg_price_point / 1000) # Assuming prices are in thousands, adjust scaling

        final_scorecard_data.append({
            'Channel Username': channel_username,
            'Channel Title': df[df['Channel Username'] == channel_username]['Channel Title'].iloc[0] if not df[df['Channel Username'] == channel_username].empty else 'N/A',
            'Total Posts': data['total_posts'],
            'First Post Date': data['first_post_date'].strftime('%Y-%m-%d') if data['first_post_date'] else 'N/A',
            'Last Post Date': data['last_post_date'].strftime('%Y-%m-%d') if data['last_post_date'] else 'N/A',
            'Posting Frequency (per week)': f"{posting_frequency_per_week:.2f}",
            'Total Views': data['total_views'],
            'Average Views per Post': f"{avg_views_per_post:.2f}",
            'Average Price Point (ETB)': f"{avg_price_point:.2f}",
            'Products Listed (Unique)': ", ".join(set(data['products_listed'])) if data['products_listed'] else "N/A",
            'Top Performing Post Text (Highest Views)': data.get('max_views_post_text', 'N/A'),
            'Products in Top Post': top_post_product_str,
            'Prices in Top Post (ETB Avg)': top_post_price_str,
            'Calculated Lending Score': f"{lending_score:.2f}"
        })

    scorecard_df = pd.DataFrame(final_scorecard_data)
    scorecard_df.to_csv(VENDOR_SCORECARD_OUTPUT, index=False, encoding='utf-8')
    print(f"\nVendor Scorecard saved to: {VENDOR_SCORECARD_OUTPUT}")
    print("\n--- Vendor Scorecard Summary ---")
    print(scorecard_df[['Channel Username', 'Total Posts', 'Posting Frequency (per week)', 'Average Views per Post', 'Average Price Point (ETB)', 'Calculated Lending Score']].to_string())


if __name__ == '__main__':
    analyze_vendors()