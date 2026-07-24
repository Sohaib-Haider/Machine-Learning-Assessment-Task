from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

app = FastAPI(title="Customer Query Classification API")

# load model
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

class_names = ["World", "Sports", "Business", "Sci/Tech"]

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


# request schema
class QueryRequest(BaseModel):
    text: str

@app.post("/predict")

def predict(request: QueryRequest):

    clean_text = preprocess_text(request.text)

    text_vector = vectorizer.transform([clean_text])

    prediction = model.predict(text_vector)[0]

    category = class_names[prediction]
    
    return {"category": category}