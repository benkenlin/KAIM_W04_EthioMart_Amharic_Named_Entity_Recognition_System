import os
from datasets import load_dataset, Features, Value, Sequence, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from collections import defaultdict

# --- Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

LABELED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'labeled')
EXTERNAL_NEWS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'amharic_news_ner_dataset')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models', 'xlm_roberta_ner') # Change this for different models

# --- Configuration ---
MODEL_NAME = "xlm-roberta-base" # Consider "rasyosef/afro-xlmr-base" for better Amharic performance
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

# Define your target NER tags for EthioMart
ETHIO_MART_LABELS = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']

# The final set of labels your model will predict.
# Here, we strictly use the EthioMart labels. External data's irrelevant tags are mapped to 'O'.
FINAL_NER_LABELS = ETHIO_MART_LABELS

# Create ID to Label and Label to ID mappings
ID2LABEL = {i: label for i, label in enumerate(FINAL_NER_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(FINAL_NER_LABELS)}

# --- Custom Data Loading Function for CoNLL-like files ---
def load_and_process_conll_file(file_path, label_mapping=None):
    """
    Reads a CoNLL-like file (word \t label per line, blank line for sentence separation)
    and converts it into a list of dictionaries suitable for Hugging Face Datasets.
    Handles -DOCSTART- lines.
    
    Args:
        file_path (str): Path to the CoNLL-like file.
        label_mapping (dict, optional): A dictionary to map original labels to new labels.
                                        If a label is not in the mapping, it defaults to 'O'.
                                        Defaults to None (no mapping, use original labels).
    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """
    data_items = []
    tokens = []
    ner_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of sentence
                if tokens: # Ensure there are tokens before adding a sentence
                    # Apply mapping if provided, otherwise use original tags
                    mapped_ner_tags = [label_mapping.get(tag, 'O') if label_mapping else tag for tag in ner_tags]
                    data_items.append({'tokens': tokens, 'ner_tags': mapped_ner_tags})
                    tokens = []
                    ner_tags = []
            elif line.startswith("-DOCSTART-"): # Skip document start markers
                continue
            else:
                parts = line.split('\t') # Assume tab-separated
                if len(parts) == 2:
                    tokens.append(parts[0])
                    ner_tags.append(parts[1])
                elif len(parts) > 2: # Handle cases with more columns, take first two
                    tokens.append(parts[0])
                    ner_tags.append(parts[1])
                # else: Skip malformed lines (e.g., lines with only one part, or comments without tabs)
        if tokens: # Add the last sentence if file doesn't end with blank line
            mapped_ner_tags = [label_mapping.get(tag, 'O') if label_mapping else tag for tag in ner_tags]
            data_items.append({'tokens': tokens, 'ner_tags': mapped_ner_tags})
    
    # Define features for the dataset consistent with NER task
    features = Features({
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(Value('string'))
    })
    
    # Create and return a datasets.Dataset object
    return load_dataset('json', data_files={'data': data_items}, features=features)['data']


# --- Main Data Loading Function ---
def load_all_datasets():
    # EthioMart Data (your manually labeled data)
    print("Loading EthioMart labeled data...")
    ethio_mart_train = load_and_process_conll_file(os.path.join(LABELED_DATA_DIR, "train.conll"))
    ethio_mart_dev = load_and_process_conll_file(os.path.join(LABELED_DATA_DIR, "dev.conll"))
    ethio_mart_test = load_and_process_conll_file(os.path.join(LABELED_DATA_DIR, "test.conll"))
    print(f"EthioMart Train: {len(ethio_mart_train)} sentences, Dev: {len(ethio_mart_dev)} sentences, Test: {len(ethio_mart_test)} sentences.")

    # Amharic News NER Data (external data)
    print("Loading external Amharic News NER data...")

    # Define a mapping for external labels to your target labels
    # Verify the actual tags in the external dataset's files (train.txt, dev.txt, test.txt)
    # Based on https://github.com/uhh-lt/ethiopicmodels/blob/master/am/data/NER/train.txt
    # Common external tags: LOC, ORG, PER, TIME, DATE, EVENT, QUANTITY, GPE, NORP, FAC, MISC, MONEY, LAW, WORK_OF_ART
    external_to_ethiomart_label_map = defaultdict(lambda: 'O', {
        'O': 'O',
        'B-LOC': 'B-LOC', 'I-LOC': 'I-LOC',
        'B-MONEY': 'B-PRICE', 'I-MONEY': 'I-PRICE',
        'B-QUANTITY': 'B-PRICE', 'I-QUANTITY': 'I-PRICE',
        # Other news-specific tags (PER, ORG, TIME, DATE, EVENT, etc.) are mapped to 'O'
        # Add 'B-PRODUCT', 'I-PRODUCT' if the external dataset explicitly has them and you want to use them.
        # This dataset is for news, so 'PRODUCT' tags are unlikely to be present.
    })

    news_train = load_and_process_conll_file(os.path.join(EXTERNAL_NEWS_DATA_DIR, "train.txt"), label_mapping=external_to_ethiomart_label_map)
    news_dev = load_and_process_conll_file(os.path.join(EXTERNAL_NEWS_DATA_DIR, "dev.txt"), label_mapping=external_to_ethiomart_label_map)
    news_test = load_and_process_conll_file(os.path.join(EXTERNAL_NEWS_DATA_DIR, "test.txt"), label_mapping=external_to_ethiomart_label_map)
    print(f"News Train: {len(news_train)} sentences, Dev: {len(news_dev)} sentences, Test: {len(news_test)} sentences.")

    # Combine datasets
    print("Combining datasets...")
    # Combine EthioMart train with News train for a larger training set.
    combined_train_dataset = ethio_mart_train.add_batch(news_train)
    
    # For validation, prioritize your domain-specific data.
    # If EthioMart dev is very small, you might add some from news_dev,
    # but be cautious about validation metrics becoming too general.
    combined_validation_dataset = ethio_mart_dev # Primary validation set
    
    # Always keep your EthioMart test set separate for true domain-specific evaluation.
    final_test_dataset = ethio_mart_test

    print(f"Combined Train: {len(combined_train_dataset)} sentences.")

    return DatasetDict({
        'train': combined_train_dataset,
        'validation': combined_validation_dataset,
        'test': final_test_dataset
    })


# --- Tokenization and Label Alignment ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Ensure the label exists in LABEL2ID, otherwise assign -100
                label_ids.append(LABEL2ID.get(label[word_idx], -100))
            else:
                label_ids.append(-100) # Only label the first token of a word
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --- Compute Metrics ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (token with -100 label)
    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [[ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    # Use seqeval for standard NER metrics
    # Ensure that `true_labels` and `true_predictions` do not contain empty lists if a sentence had no entities after filtering -100
    # seqeval can handle empty lists.
    report = classification_report(true_labels, true_predictions, output_dict=True)

    # seqeval's classification_report might structure "overall accuracy" differently.
    # It provides per-entity metrics and a "micro avg". 'accuracy' in its output refers to token-level accuracy.
    # For NER, F1-score for entities (micro avg) is usually the primary metric.
    
    return {
        "precision": report["micro avg"]["precision"] if "micro avg" in report else 0.0,
        "recall": report["micro avg"]["recall"] if "micro avg" in report else 0.0,
        "f1": report["micro avg"]["f1"] if "micro avg" in report else 0.0,
        "accuracy": report["accuracy"] if "accuracy" in report else 0.0 # This is token-level accuracy
    }

# --- Main Training Logic ---
def train_ner_model():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(FINAL_NER_LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    print("Loading and combining labeled datasets...")
    raw_datasets = load_all_datasets()
    
    print("Tokenizing and aligning labels for combined dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        # Remove original columns from each split (e.g., 'tokens', 'ner_tags')
        remove_columns=raw_datasets["train"].column_names # Assumes all splits have same original columns
    )
    
    # Data Collator (for padding and converting to PyTorch tensors)
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    print("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=10,
        save_strategy="epoch", # Save model at the end of each epoch
        load_best_model_at_end=True, # Load the best model found during training on evaluation set
        metric_for_best_model="f1", # Metric to determine the best model
        report_to="none" # Or "tensorboard" if you set it up
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer, # Pass tokenizer to Trainer for automatic saving
        compute_metrics=compute_metrics,
    )

    print("Starting model training...")
    trainer.train()

    print("\nTraining complete. Evaluating on EthioMart test set...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print("Test Results:", results)

    # Save the best model
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == '__main__':
    train_ner_model()