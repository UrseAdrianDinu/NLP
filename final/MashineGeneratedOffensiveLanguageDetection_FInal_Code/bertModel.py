import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from preprocessing import preprocess_data  # Importing the preprocessing functions
import os

# Load your new dataset
df = pd.read_csv("final_human_gen_comb_df.csv")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Rename the columns for consistency with preprocessing function
df.rename(columns={'tweet': 'text'}, inplace=True)

# Preprocess the text data
df = preprocess_data(df)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['human'])

# Tokenization and Input Formatting
BATCH_SIZE = 16
MAX_LEN = 256
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',  # Use padding='max_length' instead of pad_to_max_length
            truncation=True,       # Ensure truncation happens to max length
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

# Prepare training and validation data
y_train = train_df['human']
X_train = train_df['cleaned_text']
y_valid = test_df['human']
X_valid = test_df['cleaned_text']

# Encode data
train_inputs, train_masks = bert_encode(X_train, tokenizer)
validation_inputs, validation_masks = bert_encode(X_valid, tokenizer)

# Convert labels to torch tensors
train_labels = torch.tensor(y_train.values)
validation_labels = torch.tensor(y_valid.values)

# Create TensorDatasets
train_data = TensorDataset(train_inputs, train_masks, train_labels)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

# Create DataLoaders
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=BATCH_SIZE)

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 5

# Initialize metrics storage
train_loss_set = []
validation_accuracy = []
validation_precision = []
validation_recall = []
validation_f1 = []

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    train_loss_set.append(avg_train_loss)
    print(f"Epoch {epoch + 1}: Average train loss: {avg_train_loss}")

    # Validation step
    model.eval()
    preds, true_labels = [], []
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    eval_accuracy = np.sum(np.array(preds) == np.array(true_labels)) / len(true_labels)
    eval_precision = precision_score(true_labels, preds, average='macro')
    eval_recall = recall_score(true_labels, preds, average='macro')
    eval_f1 = f1_score(true_labels, preds, average='macro')

    validation_accuracy.append(eval_accuracy)
    validation_precision.append(eval_precision)
    validation_recall.append(eval_recall)
    validation_f1.append(eval_f1)

    print(f"Epoch {epoch + 1}: Validation Accuracy: {eval_accuracy:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, F1 Score: {eval_f1:.4f}")

# Save the trained model
model_save_path = os.path.join(os.getcwd(), 'offensive_language_detection_model.pth')
torch.save(model.state_dict(), model_save_path)

# Predictions on test data
test_inputs, test_masks = bert_encode(test_df['cleaned_text'], tokenizer)
test_data = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

model.eval()
predictions = []
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    predictions.extend(np.argmax(logits, axis=1).flatten())

# Add predictions to the test dataframe
test_df['predictions'] = predictions

# Final evaluation on test data
test_labels = test_df['human'].values
test_accuracy = np.sum(np.array(predictions) == test_labels) / len(test_labels)
test_precision = precision_score(test_labels, predictions, average='macro')
test_recall = recall_score(test_labels, predictions, average='macro')
test_f1 = f1_score(test_labels, predictions, average='macro')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Plotting the metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(range(1, epochs+1), train_loss_set, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, epochs+1), validation_accuracy, label='Validation Accuracy', color='b')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, epochs+1), validation_precision, label='Validation Precision', color='r')
plt.title('Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, epochs+1), validation_recall, label='Validation Recall', color='g')
plt.title('Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()

# Adding F1 Score Plot
plt.figure()
plt.plot(range(1, epochs+1), validation_f1, label='Validation F1 Score', color='purple')
plt.title('Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Print the test dataframe with predictions
test_df.rename(columns={'text': 'tweet'}, inplace=True)  # Rename the column back for printing
print(test_df[['tweet', 'predictions']])
