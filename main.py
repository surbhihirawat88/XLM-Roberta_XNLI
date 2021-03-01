import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

from time import time
from PIL import Image
from zipfile import ZipFile
import os, sys, itertools, re
import warnings, pickle, string
from flask import Flask, request, render_template

import torch
from transformers import XLMRobertaTokenizer


# Download the tokenizer for the XLM-Robert `base` model.
xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base" )

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# Specify our batch size.
batch_size = 16


from transformers import XLMRobertaForSequenceClassification
import torch


PATH = "xlm.pt"

def Prediction(sentence_1,sentence_2):
  tokenizer1=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
  model1=torch.load(PATH,map_location='cpu')
  line_tokenized=tokenizer1.batch_encode_plus(sentence_1,sentence_2)
  input_ids1=line_tokenized["input_ids"]
  attention_masks1=line_tokenized["attention_mask"]
  prediction_inputs1=torch.tensor(input_ids1)
  prediction_masks1=torch.tensor(attention_masks1)
  prediction_data1=TensorDataset(prediction_inputs1,prediction_masks1)
  prediction_dataloader1= DataLoader(prediction_data1,  batch_size=1)
  # Put model in evaluation mode
  model1.eval()
# Tracking variables
  predictions = []
  true_labels=[]
# Predict
  for batch in prediction_dataloader1:
  # Add batch to GPU
    batch = tuple(t.to('cpu') for t in batch)
  # Unpack the inputs from our dataloader
    b_input_ids, b_input_masks = batch
  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs1 = model1(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_masks)
    logits1 = outputs1[0]

  # Move logits and labels to CPU
    logits1 = logits1.detach().cpu().numpy()

  # Store predictions and true labels
    predictions.append(logits1)
    flat_predictions = np.concatenate(predictions, axis=0)
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

    return predicted_labels

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        question_1 = request.form['sentence_1']
        question_2= request.form['sentence_2']
        pred=Prediction([question_1],[question_2])
        return render_template('index.html', data=pred)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app