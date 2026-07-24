import re
import string
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

#load Dataset
dataset = load_dataset("fancyzhx/ag_news")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

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
train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

print("Preprocessing Done ... \n")

print(train_df[['text', 'clean_text']].head())
print(test_df[['text', 'clean_text']].head())


#Prepare labels
X_train, y_train = train_df['clean_text'], train_df['label']

X_test, y_test = test_df['clean_text'], test_df['label']

# Vectorization
print("Vectorizing text... \n")

vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  


# Model training
print("Training model... \n")

model = LogisticRegression(max_iter=1000, n_jobs=-1)

model.fit(X_train_vec, y_train)


# Evaluation
print("Evaluating model... \n")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

class_names = dataset["train"].features["label"].names
print(classification_report(y_test, y_pred, target_names=class_names))
