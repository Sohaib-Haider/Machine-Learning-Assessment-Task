# Customer Query Classification

A machine learning pipeline that classifies customer queries into 4 categories
(World, Sports, Business, Sci/Tech) and serves predictions via a REST API.

## Project Structure

- `eda.ipynb` — Notebook where I explored the data first
- `train.py` — Cleans the data, trains the model, checks accuracy, saves everything
- `api.py` — The API that loads the model and makes predictions
- `requirements.txt` — List of packages needed to run this
- `models/`
  - `model.joblib` — The trained model
  - `vectorizer.joblib` — The tool that turns text into numbers

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Sohaib-Haider/Machine-Learning-Assessment-Task.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (this also saves `models/model.joblib` and `models/vectorizer.joblib`):
```bash
python train.py
```

4. Start the API:
```bash
uvicorn api:app --reload
```

5. Test it at `http://127.0.0.1:8000/docs` (interactive Swagger UI), or send a request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"NASA launches new satellite into orbit\"}"
```
Expected response:
```json
{"category": "Sci/Tech"}
```

## Dataset

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) — 120,000 training
articles and 7,600 test articles, evenly split across 4 balanced classes
(30,000 each in training): World, Sports, Business, Sci/Tech.

## Methodology

### Preprocessing
- Converted all text to lowercase
- Removed URLs (anything starting with `http` or `www`)
- Stripped out all punctuation
- Collapsed extra/multiple spaces into one

This keeps the text in a consistent format so the model isn't confused by
superficial differences (e.g. "Oil" vs "oil") that don't affect meaning.
No missing values were found in the dataset during EDA, so no imputation
was needed.

### Vectorization
- Used **TF-IDF** (`TfidfVectorizer`) to convert text into numeric features
- Capped at **20,000 max features** to keep training fast and memory-efficient
- Stopwords (e.g. "the", "is", "and") removed here using `stop_words='english'`, rather than inside the preprocessing function

**Why stopwords weren't removed in `preprocess_text()`:** This was a
deliberate choice, not an oversight. Removing stopwords was left to
`TfidfVectorizer(stop_words='english')` instead, so each part of the
pipeline has one clear job — `preprocess_text()` only standardizes the raw
text format (case, URLs, punctuation, spacing), while the vectorizer handles
everything related to turning words into model-ready features, including
which words to ignore. This also means the vectorizer's built-in, well-tested
English stopword list is used, rather than maintaining a separate custom one.

### Model
- **Logistic Regression** was chosen over Naive Bayes and Random Forest
- TF-IDF produces sparse, high-dimensional features, which Logistic Regression handles well
- Random Forest is generally less suited to sparse text data
- Logistic Regression is also fast to train and easy to interpret, making it a strong standard baseline for text classification

### A note on typos
Typos were not specifically handled, since AG News is professionally-written
news text (Reuters/AP), making misspellings rare. The 20k max-feature cap
also naturally filters out most rare/misspelled tokens.

## Evaluation

The dataset's built-in train/test split (120,000 / 7,600) was used rather than
a manual split, since the dataset already provides one intended for this
purpose.

**Accuracy: 91.46%**

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| World     | 0.93      | 0.90   | 0.92     |
| Sports    | 0.95      | 0.98   | 0.97     |
| Business  | 0.88      | 0.88   | 0.88     |
| Sci/Tech  | 0.89      | 0.89   | 0.89     |

Per-class Precision/Recall/F1 were reported alongside overall accuracy because
accuracy alone can hide a weak-performing class behind a good overall number.
Since the classes are perfectly balanced (30,000 samples each, confirmed in
EDA), accuracy is a fair primary metric here, and macro/weighted averages
align closely — both around 0.91, confirming no hidden imbalance is
distorting the score.

Sports is the easiest class to classify (F1 0.97), likely due to distinctive
vocabulary (team names, scores, players). Business and Sci/Tech show the most
confusion with each other (F1 0.88–0.89), which is expected since news about
tech companies often discusses financial/business topics (earnings, IPOs,
stock prices), creating vocabulary overlap between the two categories.