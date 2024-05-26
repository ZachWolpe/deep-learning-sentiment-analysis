"""
----------------------------------------------------------------------------------
Roberta_classifier.py

Roberta classifier code.

: 25.05.24
: zachcolinwolpe@medibio.com
----------------------------------------------------------------------------------
"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
from transformers import RobertaModel 
import torch


# Define the model with activation function and regularization
class RobertaWithActivationAndRegularization(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_prob=0.5, weight_decay=0.01):
        super(RobertaWithActivationAndRegularization, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.activation = nn.GELU()
        self.weight_decay = weight_decay
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.activation(logits)  # Applying GELU activation function
        return logits


def RoBertaTokenizer(X, y): 
    """
        : Tokenize the input data using the RoBERTa tokenizer.
        : Transform to torch.tensors
        : Return the tokenized inputs, attention masks, and y values
    """
    # Tokenize the "Review Text" column
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenized_inputs = tokenizer(X['Review Text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = tokenized_inputs.input_ids
    attention_masks = tokenized_inputs.attention_mask
    y = torch.tensor(y['Sentiment'].tolist())
    return input_ids, attention_masks, y

    