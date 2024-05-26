"""
----------------------------------------------------------------------------------
ML_training_code.py

Training architecture code.

: 25.05.24
: zachcolinwolpe@medibio.com
----------------------------------------------------------------------------------
"""
import logging

# importing neccessary libraries 
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel,AdamW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd 
import numpy as np
import torch
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import sklearn


from nltk.corpus import stopwords
import argparse
import logging
import string
import re
import os

from sklearn.model_selection import train_test_split, KFold
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import BertModel
import os



def generate_K_Fold_data(X_train, y_train, num_splits=10):
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        logging.info(f"Fold {fold+1}/{num_splits}")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        yield X_train_fold, y_train_fold, X_val_fold, y_val_fold, train_index, val_index


def torch_tensorize(
        input_ids_train,
        attention_masks_train,
        y_train_fold,
        input_ids_val,
        attention_masks_val,
        y_val_fold,
        BATCH_SIZE=32
        ):
    # Create PyTorch datasets and data loaders for this fold
    if torch.cuda.is_available():
        input_ids_train = input_ids_train.cuda()
        input_ids_val = input_ids_val.cuda()
        attention_masks_train = attention_masks_train.cuda()
        attention_masks_val = attention_masks_val.cuda()
        y_train_fold = torch.tensor(y_train_fold.values).cuda()
        y_val_fold = torch.tensor(y_val_fold.values).cuda()

    # tensorize the data
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, torch.tensor(y_train_fold.values))
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, torch.tensor(y_val_fold.values))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return (
        train_dataloader,
        train_dataloader,
        val_dataset,
        val_dataloader)


def training_loop(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        device,
        epochs=6):
    print('Training the model...')
    # Lists to store losses for this fold
    train_losses_fold = []
    valid_losses_fold = []
    accuracies_fold = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            labels = labels.reshape(-1)  # Reshape labels once

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            loss = criterion(logits, labels)
            total_train_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses_fold.append(average_train_loss)

        # Validation loop
        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                labels = labels.reshape(-1)  # Reshape labels once

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                loss = criterion(logits, labels)
                total_valid_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct_valid += (predicted == labels).sum().item() 
                total_valid += labels.size(0)

        average_valid_loss = total_valid_loss / len(val_dataloader)
        valid_losses_fold.append(average_valid_loss)

        accuracy_train = correct_train / total_train
        accuracy_valid = correct_valid / total_valid
        accuracies_fold.append((accuracy_train, accuracy_valid))

        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {average_train_loss:.4f} - Validation Loss: {average_valid_loss:.4f} - Training Accuracy: {accuracy_train:.4f} - Validation Accuracy: {accuracy_valid:.4f}')

    print('Finished Training.')
    return train_losses_fold, valid_losses_fold, accuracies_fold



def plot_training_validation(train_losses_fold, valid_losses_fold):
    # Plotting training and validation losses for all folds
    num_epochs = len(train_losses_fold)
    plt.figure(figsize=(4, 2))
    plt.plot(range(1, num_epochs + 1), train_losses_fold, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses_fold, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs for all Folds')
    plt.legend()
    plt.grid(True)
    plt.show()

