import asyncio
import csv
import os
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument # Import for document handling
import pandas as pd # For reading channels_to_crawl.xlsx
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
# It's crucial to specify the path to your .env file
# Assuming .env is in the config/ directory relative to your project root
# And you're running this script from the project root (e.g., python scripts/telegram_scrapper.py)
load_dotenv('config/.env') # Changed this path!

api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('TG_PHONE_NUMBER') # Renamed for clarity and consistency

# --- Basic Validation of Credentials ---
if not all([api_id, api_hash, phone]):
    print("Error: Telegram API credentials (TG_API_ID, TG_API_HASH, TG_PHONE_NUMBER) not found in config/.env file.")
    print("Please ensure your .env file is correctly set up and contains these variables.")
    exit(1) # Exit if credentials are missing

# --- 2. Define File Paths (Aligned with Project Structure) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the current script
# Adjust PROJECT_ROOT to be the actual root if this script is nested further
# In your case, if telegram_scrapper.py is in scripts/, then PROJECT_ROOT will be scripts/
# We need to go up one level to reach the main project root
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir)) # Go up from 'scripts'
# Now PROJECT_ROOT is 'KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System'

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMAGES_DIR = os.path.join(DATA_RAW_DIR, 'images')
DOCUMENTS_DIR = os.path.join(DATA_RAW_DIR, 'documents') # New directory for documents
CSV_OUTPUT_FILE = os.path.join(DATA_RAW_DIR, 'telegram_data.csv') # Sticking with CSV for now, will convert to JSONL later
CHANNELS_LIST_FILE = os.path.join(DATA_RAW_DIR, 'channels_to_crawl.xlsx') # Path to your channel list

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True) # Create documents directory

# --- 3. Function to Scrape Data from a Single Channel ---
async def scrape_channel(client, channel_username, writer, images_dir, documents_dir, limit=10000):
    """
    Scrapes messages from a single Telegram channel, including text and media,
    and writes to a CSV file.
    """
    print(f"Attempting to scrape from channel: {channel_username}")
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        # Get channel views count if available (for vendor scorecard)
        # full_channel = await client(GetFullChannelRequest(entity.id)) # Requires GetFullChannelRequest import
        # channel_views = full_channel.full_chat.participants_count # Or other relevant stats

        async for message in client.iter_messages(entity, limit=limit):
            media_path = None
            message_type = 'text' # Default

            if message.media:
                if isinstance(message.media, MessageMediaPhoto):
                    message_type = 'photo'
                    filename = f"{channel_username}_{message.id}.jpg"
                    media_path = os.path.join(images_dir, filename)
                    await client.download_media(message.media, file=media_path)
                    # print(f"  Downloaded photo: {filename}")
                elif isinstance(message.media, MessageMediaDocument):
                    message_type = 'document'
                    # Get original file name or use a generic one
                    doc_filename = message.media.document.attributes[0].file_name if message.media.document.attributes else f"{channel_username}_{message.id}.bin"
                    media_path = os.path.join(documents_dir, doc_filename)
                    await client.download_media(message.media, file=media_path)
                    # print(f"  Downloaded document: {doc_filename}")
                # Add more media types if needed (e.g., videos, voice messages)

            # Extract views (if available in the message object)
            views_count = message.views if hasattr(message, 'views') else None

            writer.writerow([
                channel_title,
                channel_username,
                message.id,
                message.message, # This is the message text (None if only media)
                message.date.isoformat(), # Convert datetime to ISO string for easier storage
                media_path,
                message_type,
                views_count
            ])
            # Small delay to be polite and avoid rate limits
            await asyncio.sleep(0.1)

        print(f"Finished scraping {limit} messages (or all available) from {channel_username}.")

    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")
    finally:
        # A small delay after each channel to cool down the connection
        await asyncio.sleep(2)


# --- 4. Main Asynchronous Function to Orchestrate Scraping ---
async def main():
    # Initialize the Telegram client
    client = TelegramClient('scraping_session', api_id, api_hash)
    print("Connecting to Telegram...")
    try:
        await client.connect()

        if not await client.is_user_authorized():
            print("First time connecting. Please enter your phone number and verification code.")
            await client.start(phone=phone)
        print("Connected to Telegram successfully.")

        # Read channels from the Excel file
        try:
            df_channels = pd.read_excel(CHANNELS_LIST_FILE)
            # Assuming the channel usernames are in a column named 'Username' or similar
            # Adjust 'Channel Username' to your actual column name in channels_to_crawl.xlsx
            channels_to_scrape = df_channels['Channel Username'].dropna().tolist()
            if not channels_to_scrape:
                print(f"Warning: No channel usernames found in '{CHANNELS_LIST_FILE}'. Please check the 'Channel Username' column.")
                return # Exit if no channels to scrape
            print(f"Loaded {len(channels_to_scrape)} channels from '{CHANNELS_LIST_FILE}'.")
            print("Channels:", channels_to_scrape)

        except FileNotFoundError:
            print(f"Error: Channels list file '{CHANNELS_LIST_FILE}' not found.")
            print("Please create 'data/raw/channels_to_crawl.xlsx' with a 'Channel Username' column.")
            return # Exit if channels file is missing
        except KeyError:
            print(f"Error: 'Channel Username' column not found in '{CHANNELS_LIST_FILE}'.")
            print("Please ensure the Excel file has a column named 'Channel Username' (case-sensitive).")
            return # Exit if column is missing


        # Open the CSV file in write mode
        with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Include 'Media Type' and 'Views' in the header
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path', 'Media Type', 'Views'])

            # Iterate over channels and scrape data
            for channel_username in channels_to_scrape:
                await scrape_channel(client, channel_username, writer, IMAGES_DIR, DOCUMENTS_DIR)

    except Exception as e:
        print(f"An error occurred during the main scraping process: {e}")
    finally:
        print("Disconnecting from Telegram.")
        await client.disconnect()

# --- 5. Run the main function ---
if __name__ == '__main__':
    # Use client context manager to ensure proper disconnection
    # This structure is a bit more robust for client lifecycle management
    asyncio.run(main()) # Changed to asyncio.run(main())