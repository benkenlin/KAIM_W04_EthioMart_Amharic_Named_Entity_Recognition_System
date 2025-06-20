import asyncio
import csv
import os
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import pandas as pd
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv('config/.env')

api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('TG_PHONE_NUMBER')

if not all([api_id, api_hash, phone]):
    print("Error: Telegram API credentials (TG_API_ID, TG_API_HASH, TG_PHONE_NUMBER) not found in config/.env file.")
    print("Please ensure your .env file is correctly set up and contains these variables.")
    exit(1)

# --- Define File Paths ---
# Get the directory of the current script (scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root (e.g., KAIM_W04_EthioMart_Amharic_Named_Entity_Recognition_System)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMAGES_DIR = os.path.join(DATA_RAW_DIR, 'images')
DOCUMENTS_DIR = os.path.join(DATA_RAW_DIR, 'documents')
CSV_OUTPUT_FILE = os.path.join(DATA_RAW_DIR, 'telegram_data.csv')
CHANNELS_LIST_FILE = os.path.join(DATA_RAW_DIR, 'channels_to_crawl.xlsx')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# --- Function to Scrape Data from a Single Channel ---
async def scrape_channel(client, channel_username, writer, images_dir, documents_dir, limit=None):
    """
    Scrapes messages from a single Telegram channel, including text and media,
    and writes to a CSV file.
    """
    print(f"Attempting to scrape from channel: {channel_username}")
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title

        #Decide on a sensible default limit if not provided
        # For typical use cases, getting the latest few thousand messages is often sufficient.
        # You can adjust this number based on your needs.
        effective_limit = limit if limit is not None else 5000 # Example: default to latest 5000 messages

        message_count = 0
        async for message in client.iter_messages(entity, limit=effective_limit):
            message_count += 1
            # [DEBUG] line can be removed now, or kept for verbose output
            # print(f"  [DEBUG - {channel_username}] Scraped message ID: {message.id}, Date: {message.date}, Text snippet: '{message.message[:50] if message.message else 'N/A'}'")

            media_path = None
            message_type = 'text' # Default

            if message.media:
                if isinstance(message.media, MessageMediaPhoto):
                    message_type = 'photo'
                    filename = f"{channel_username}_{message.id}.jpg"
                    # Corrected path for media saving, ensure it saves inside IMAGES_DIR
                    media_path = os.path.join(images_dir, filename)
                    # The download_media method directly handles the path provided
                    await client.download_media(message.media, file=media_path)
                elif isinstance(message.media, MessageMediaDocument):
                    message_type = 'document'
                    doc_filename = message.media.document.attributes[0].file_name if message.media.document.attributes else f"{channel_username}_{message.id}.bin"
                    # Corrected path for media saving, ensure it saves inside DOCUMENTS_DIR
                    media_path = os.path.join(documents_dir, doc_filename)
                    await client.download_media(message.media, file=media_path)

            views_count = message.views if hasattr(message, 'views') else None
            message_text = message.message if message.message else ""

            writer.writerow([
                channel_title,
                channel_username,
                message.id,
                message_text,
                message.date.isoformat(),
                media_path,
                message_type,
                views_count
            ])
            await asyncio.sleep(0.1) # Small delay

        print(f"Finished scraping messages from {channel_username}. Total messages found and written: {message_count}")
        if message_count == 0:
            print(f"  [WARNING] No messages were found for channel {channel_username}. This channel might be empty or inaccessible within the given limit.")

    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")
    finally:
        await asyncio.sleep(2) # Longer delay after each channel


# --- Main Asynchronous Function to Orchestrate Scraping ---
async def main():
    client = TelegramClient('scraping_session', api_id, api_hash, timeout=30) # Default is 10 seconds, try 30 or 60
    print("Connecting to Telegram...")
    try:
        await client.connect()

        if not await client.is_user_authorized():
            print("First time connecting. Please enter your phone number and verification code.")
            await client.start(phone=phone)
        print("Connected to Telegram successfully.")

        try:
            df_channels = pd.read_excel(CHANNELS_LIST_FILE)
            channels_to_scrape = df_channels['Channel Username'].dropna().tolist()
            if not channels_to_scrape:
                print(f"Warning: No channel usernames found in '{CHANNELS_LIST_FILE}'. Please check the 'Channel Username' column.")
                return
            print(f"Loaded {len(channels_to_scrape)} channels from '{CHANNELS_LIST_FILE}'.")

        except FileNotFoundError:
            print(f"Error: Channels list file '{CHANNELS_LIST_FILE}' not found.")
            print("Please create 'data/raw/channels_to_crawl.xlsx' with a 'Channel Username' column.")
            return
        except KeyError:
            print(f"Error: 'Channel Username' column not found in '{CHANNELS_LIST_FILE}'.")
            print("Please ensure the Excel file has a column named 'Channel Username' (case-sensitive).")
            return


        with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path', 'Media Type', 'Views'])

            for channel_username in channels_to_scrape:
                await scrape_channel(client, channel_username, writer, IMAGES_DIR, DOCUMENTS_DIR, limit=None) # Set limit=5000 or None for all

    except Exception as e:
        print(f"An error occurred during the main scraping process: {e}")
    finally:
        print("Disconnecting from Telegram.")
        await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())