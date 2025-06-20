import os
import random

# Define file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

LABELED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'labeled')
RAW_LABELED_FILE = os.path.join(LABELED_DATA_DIR, 'labeled_telegram_product_price_location.txt') # Your input file

TRAIN_FILE = os.path.join(LABELED_DATA_DIR, 'train.conll')
DEV_FILE = os.path.join(LABELED_DATA_DIR, 'dev.conll')
TEST_FILE = os.path.join(LABELED_DATA_DIR, 'test.conll')

# Define split ratios
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1 # Test ratio will be 1 - TRAIN_RATIO - DEV_RATIO
# For a small dataset (30-50 messages), these ratios might result in very few samples per set.
# You might consider 80/20 train/dev and use dev as test, or just 90/10 if data is very limited.

def read_conll_file(filepath):
    """Reads a CoNLL file and returns a list of sentences, where each sentence
    is a list of (token, label) tuples."""
    sentences = []
    current_sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) == 2: # Expecting "token label" format
                    current_sentence.append((parts[0], parts[1]))
                else:
                    print(f"Warning: Skipping malformed line in CoNLL file: {line}")
        if current_sentence: # Add last sentence if file doesn't end with blank line
            sentences.append(current_sentence)
    return sentences

def write_conll_file(filepath, sentences):
    """Writes a list of sentences (in (token, label) format) to a CoNLL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n") # Blank line between sentences

def split_data(input_file, train_out, dev_out, test_out, train_ratio, dev_ratio):
    print(f"Loading data from {input_file}...")
    sentences = read_conll_file(input_file)
    print(f"Loaded {len(sentences)} sentences.")

    # Shuffle sentences to ensure random distribution
    random.seed(42) # For reproducibility
    random.shuffle(sentences)

    total_sentences = len(sentences)
    train_split = int(total_sentences * train_ratio)
    dev_split = int(total_sentences * dev_ratio)

    train_sentences = sentences[:train_split]
    dev_sentences = sentences[train_split : train_split + dev_split]
    test_sentences = sentences[train_split + dev_split :]

    print(f"Splitting into: Train ({len(train_sentences)}), Dev ({len(dev_sentences)}), Test ({len(test_sentences)})")

    write_conll_file(train_out, train_sentences)
    write_conll_file(dev_out, dev_sentences)
    write_conll_file(test_out, test_sentences)

    print(f"Data split and saved to:\n  Train: {train_out}\n  Dev: {dev_out}\n  Test: {test_out}")

if __name__ == '__main__':
    if not os.path.exists(RAW_LABELED_FILE):
        print(f"Error: Labeled data file not found at '{RAW_LABELED_FILE}'. Please ensure you have labeled data.")
    else:
        split_data(RAW_LABELED_FILE, TRAIN_FILE, DEV_FILE, TEST_FILE, TRAIN_RATIO, DEV_RATIO)