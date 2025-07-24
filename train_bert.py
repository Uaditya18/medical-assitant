# train_bert_model.py

import pandas as pd
import numpy as np
import torch
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Paths
dataset_path = 'Diseases_Symptoms (1)(in).csv'
model_save_dir = 'bert_model_disease_prediction'
label_encoder_path = 'label_encoder_disease.pkl'

# Ensure save directory exists
os.makedirs(model_save_dir, exist_ok=True)

# Load dataset
try:
    data = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded with {len(data)} records")
except FileNotFoundError:
    print(f"❌ Dataset file not found at {dataset_path}")
    exit(1)

# Ensure required columns exist
if 'Symptoms' not in data.columns or 'Name' not in data.columns:
    print("❌ Dataset must contain 'Symptoms' and 'Name' columns.")
    exit(1)

# Group similar diseases to reduce number of classes
data['Name'] = data['Name'].replace({
    'Ethylene glycol poisoning-1': 'Ethylene glycol poisoning',
    'Ethylene glycol poisoning-2': 'Ethylene glycol poisoning',
    'Ethylene glycol poisoning-3': 'Ethylene glycol poisoning',
    'Open-Angle Glaucoma': 'Glaucoma',
    'Angle-Closure Glaucoma': 'Glaucoma',
    'Normal-Tension Glaucoma': 'Glaucoma',
    'Congital Glaucoma': 'Glaucoma',
    'Secondary Glaucoma': 'Glaucoma',
    'Pigmentary Glaucoma': 'Glaucoma',
    'Exfoliation Glaucoma': 'Glaucoma',
    'Low-Tension Glaucoma': 'Glaucoma'
})

# Simple data augmentation with synonym replacement
synonym_map = {
    'palpitations': 'heart pounding',
    'sweating': 'perspiration',
    'fatigue': 'tiredness',
    'pain': 'discomfort',
    'swelling': 'edema'
}

def augment_symptoms(symptoms):
    augmented = symptoms
    for original, synonym in synonym_map.items():
        augmented = augmented.replace(original, synonym)
    return augmented

# Apply augmentation to create additional samples
augmented_data = data.copy()
augmented_data['Symptoms'] = augmented_data['Symptoms'].apply(augment_symptoms)
data = pd.concat([data, augmented_data], ignore_index=True)
print(f"✅ Augmented dataset size: {len(data)} records")

# Prepare labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['Name'])
num_labels = len(label_encoder.classes_)
print(f"✅ Found {num_labels} unique disease names")

# Save label encoder for later use
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ Label encoder saved to {label_encoder_path}")

# Load BioBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

# Tokenize texts
MAX_LENGTH = 128
encoded_texts = tokenizer(
    data['Symptoms'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

# Create dataset
input_ids = encoded_texts['input_ids']
attention_masks = encoded_texts['attention_mask']
labels = torch.tensor(encoded_labels, dtype=torch.long)

# Split data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42
)

# Check unique classes in validation set
print(f"✅ Unique classes in validation set: {len(set(val_labels.numpy()))}")

# Create DataLoaders
batch_size = 8  # Reduced from 16 for more gradient updates
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Load BioBERT model
model = BertForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.2',
    num_labels=num_labels
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
model.to(device)

# Optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Reduced lr, added weight decay

# Scheduler
epochs = 20  # Increased from 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
print("Starting training...")
best_accuracy = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        batch_input_ids, batch_attention_masks, batch_labels = [b.to(device) for b in batch]

        model.zero_grad()

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        batch_input_ids, batch_attention_masks, batch_labels = [b.to(device) for b in batch]

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

        loss = outputs.loss
        total_eval_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        accuracy = (predictions == batch_labels).float().mean().item()
        total_eval_accuracy += accuracy

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Calculate top-5 accuracy
    top_k_accuracy = 0
    for i, logits in enumerate(outputs.logits):
        top_k_preds = torch.topk(logits, k=5, dim=0).indices
        if batch_labels[i] in top_k_preds:
            top_k_accuracy += 1
    top_k_accuracy = top_k_accuracy / len(val_dataloader.dataset)

    print(f"Validation Accuracy: {avg_val_accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Top-5 Accuracy: {top_k_accuracy:.4f}")

    if avg_val_accuracy > best_accuracy:
        best_accuracy = avg_val_accuracy
        print(f"Saving best model with accuracy: {best_accuracy:.4f}")
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)

print("\n✅ Training complete!")
print(f"✅ Best validation accuracy: {best_accuracy:.4f}")
print(f"✅ Model and tokenizer saved to {model_save_dir}")