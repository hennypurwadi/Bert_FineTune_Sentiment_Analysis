# BERT-base-uncased Fine-tuned for Sentiment Analysis

---
language: en
license: apache-2.0
datasets:
- custom
task_categories:
- text-classification
task_ids:
- sentiment-classification
---

# BERT-base-uncased Fine-tuned for Sentiment Analysis

This model is a fine-tuned version of the `bert-base-uncased` model for sentiment analysis. It is trained on a dataset of texts with six different emotions: anger, fear, joy, love, sadness, and surprise.

## Model Training Details

- **Pretrained model**: `bert-base-uncased`
- **Number of labels**: 6,  which are ` "anger": 0,  "fear": 1,  "joy": 2,  "love": 3,  "sadness": 4,  "surprise": 5`
- **Learning rate**: 2e-5
- **Epsilon**: 1e-8
- **Epochs**: 10
- **Warmup steps**: 0
- **Optimizer**: AdamW with correct_bias=False

## Dataset

The model was trained and tested on a labeled dataset from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

##To predict the sentiments on unlabeled datasets, use the predict_sentiments function provided in this repository.

## The unlabeled daataset to be predicted should have a single column named "text". 

The model is available on Hugging Face 
## Model Hub: https://huggingface.co/RinInori/bert-base-uncased_finetune_sentiments

![Image description](https://github.com/hennypurwadi/Bert_FineTune_Sentiment_Analysis/blob/main/images/SaveModel_Tokenizer_To_HuggingFace_web.jpg?raw=true)

##To load and use the model and tokenizer, use the following code:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

def predict_sentiments(model_name, tokenizer_name, input_file):

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    df = pd.read_csv(input_file)

    # Tokenize the input text
    test_inputs = tokenizer(list(df['text']), padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Make predictions
    with torch.no_grad():
        model.eval()
        outputs = model(test_inputs['input_ids'], token_type_ids=None, attention_mask=test_inputs['attention_mask'])
        logits = outputs[0].detach().cpu().numpy()
        predictions = logits.argmax(axis=-1)

    # Map the predicted labels back to their original names
    int2label = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
    predicted_labels = [int2label[p] for p in predictions]

    # Add the predicted labels to the test dataframe
    df['label'] = predicted_labels

    # Save the predictions to a file
    output_file = input_file.replace(".csv", "_predicted.csv")
    df.to_csv(output_file, index=False)

model_name = "RinInori/bert-base-uncased_finetune_sentiments"
tokenizer_name = "RinInori/bert-base-uncased_finetune_sentiments"

#Predict Unlabeled data
predict_sentiments(model_name, tokenizer_name, '/content/drive/MyDrive/DLBBT01/data/c_unlabeled/dc_America.csv')

# Load the predicted data
df_Am = pd.read_csv('/content/drive/MyDrive/DLBBT01/data/c_unlabeled/dc_America_predicted.csv')
df_Am.head()

from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load tokenizer
tokenizer_name = "RinInori/bert-base-uncased_finetune_sentiments"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

# Load dataset
input_file = '/content/drive/MyDrive/DLBBT01/data/c_unlabeled/dc_America_predicted.csv'
df_Am = pd.read_csv(input_file)

# Examine the distribution of data based on labels
sentences = df_Am.text.values
print("Distribution of data based on labels: ", df_Am.label.value_counts())

MAX_LEN = 512

# Plot the label distribution
label_count = df_Am['label'].value_counts()
plot_users = label_count.plot.pie(autopct='%1.1f%%', figsize=(4, 4))
plt.rc('axes', unicode_minus=False)
