"""
----------------------------------------------------------------------------------
Bert_classfier.py

Bert architecture code.

: 25.05.24
: zachcolinwolpe@medibio.com
----------------------------------------------------------------------------------
"""

from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch


# Define the model with activation function and regularization
class BertWithActivationAndRegularization(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_prob=0.5, weight_decay=0.01):
        super(BertWithActivationAndRegularization, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.activation = nn.GELU()
        self.weight_decay = weight_decay
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.activation(logits)  # Applying GELU activation function
        return logits

def Bert_tokenize(X, y):
    """
        : Tokenize the input data using the BERT tokenizer.
        : Transform to torch.tensors
        : Return the tokenized inputs, attention masks, and y values
    """
    # Tokenize the "Review Text" column
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(X['Review Text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = tokenized_inputs.input_ids
    attention_masks = tokenized_inputs.attention_mask
    y = torch.tensor(y['Sentiment'].tolist())
    return input_ids, attention_masks, y