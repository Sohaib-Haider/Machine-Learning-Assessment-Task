import re
import string
from datasets import load_dataset
import pandas as pd

#load Dataset
dataset = load_dataset("fancyzhx/ag_news")

df = pd.DataFrame(dataset["train"])

print("Dataset Loaded Successfully... ... \n")


#preprocessing
def preprocess_text(text):

    #lowercase all words
    text = text.lower()
    
    #remove urls
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    
    #remover punctuation
    for char in string.punctuation:
        text = text.replace(char, " ")

    #remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

print("Preprocessing ... ... \n")
df['clean_text'] = df['text'].apply(preprocess_text)
print("Preprocessing Done ... \n")

print(df[['text', 'clean_text']].head())

