import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, Sequence
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
import json
from collections import defaultdict

# --- Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

LABELED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'labeled')
# Path to your trained model (e.g., the final_model subfolder)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xlm_roberta_ner', 'final_model') # Make sure this points to your saved model

# --- Configuration (Must match training config) ---
# Ensure these labels match the ones used during training!
FINAL_NER_LABELS = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']
ID2LABEL = {i: label for i, label in enumerate(FINAL_NER_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(FINAL_NER_LABELS)}
BATCH_SIZE = 16 # Use same or smaller batch size for evaluation

# --- Data Loading (Same as in training script) ---
def load_and_process_conll_file(file_path, label_mapping=None):
    """
    Reads a CoNLL-like file and converts it into a list of dictionaries.
    Handles -DOCSTART- lines and label mapping.
    """
    data_items = []
    tokens = []
    ner_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of sentence
                if tokens:
                    mapped_ner_tags = [label_mapping.get(tag, 'O') if label_mapping else tag for tag in ner_tags]
                    data_items.append({'tokens': tokens, 'ner_tags': mapped_ner_tags})
                    tokens = []
                    ner_tags = []
            elif line.startswith("-DOCSTART-"):
                continue
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    tokens.append(parts[0])
                    ner_tags.append(parts[1])
                elif len(parts) > 2:
                    tokens.append(parts[0])
                    ner_tags.append(parts[1])
        if tokens:
            mapped_ner_tags = [label_mapping.get(tag, 'O') if label_mapping else tag for tag in ner_tags]
            data_items.append({'tokens': tokens, 'ner_tags': mapped_ner_tags})
    
    features = Features({
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(Value('string'))
    })
    
    return load_dataset('json', data_files={'data': data_items}, features=features)['data']


# --- Tokenization and Label Alignment (Same as in training script) ---
# The tokenizer path needs to point to the saved model's tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading tokenizer from {MODEL_PATH}: {e}")
    print("Please ensure your model is trained and saved correctly.")
    exit(1)


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
                label_ids.append(LABEL2ID.get(label[word_idx], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --- Compute Metrics (Same as in training script) ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [[ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "precision": report["micro avg"]["precision"] if "micro avg" in report else 0.0,
        "recall": report["micro avg"]["recall"] if "micro avg" in report else 0.0,
        "f1": report["micro avg"]["f1"] if "micro avg" in report else 0.0,
        "accuracy": report["accuracy"] if "accuracy" in report else 0.0
    }

# --- Main Evaluation Logic ---
def evaluate_model(model_path, test_file_path):
    print(f"Loading model from: {model_path}")
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure your model is trained and saved correctly.")
        return None, None

    print(f"Loading test dataset from: {test_file_path}")
    # No label mapping needed here as we assume test.conll already has FINAL_NER_LABELS
    test_dataset = load_and_process_conll_file(test_file_path)

    print("Tokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./evaluation_output", # Temporary dir for evaluation logs
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Performing evaluation...")
    metrics = trainer.evaluate()
    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=2))

    # Get detailed classification report
    predictions, labels, _ = trainer.predict(tokenized_test_dataset)
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [[ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    print("\nDetailed Classification Report:")
    detailed_report = classification_report(true_labels, true_predictions)
    print(detailed_report)

    return metrics, classification_report(true_labels, true_predictions, output_dict=True)


if __name__ == '__main__':
    metrics, detailed_report = evaluate_model(MODEL_PATH, os.path.join(LABELED_DATA_DIR, 'test.conll'))