# Customer Query Classification

A machine learning pipeline that classifies customer queries into 4 categories
(World, Sports, Business, Sci/Tech) and serves predictions via a REST API.

## Project Structure

├── models/
│   ├── model.joblib      # Trained Logistic Regression model
│   └── model.joblib      # Trained Logistic Regression model
├── api.py                # FastAPI app serving the /predict endpoint
├── eda.ipynb             # Exploratory data analysis
├── README.md        
├── requirements.txt      # Python dependencies
├── train.py              # Preprocessing, training, evaluation

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Sohaib-Haider/Machine-Learning-Assessment-Task/
cd <your-repo-folder>
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

**Preprocessing:** Text was lowercased, stripped of URLs, stripped of
punctuation, and had extra whitespace collapsed. This standardizes the text so
the model isn't confused by superficial differences (e.g. "Oil" vs "oil") that
don't affect meaning. No missing values were found in the dataset, so no
imputation was needed.

**Vectorization:** TF-IDF (`TfidfVectorizer`) was used to convert text into
numeric features, capped at 20,000 max features to keep training fast and
memory-efficient. Stopword removal (`stop_words='english'`) was handled at
this stage rather than in the custom preprocessing function, keeping each
step's responsibility clear.

**Model:** Logistic Regression was chosen over Naive Bayes and Random Forest.
TF-IDF produces sparse, high-dimensional features, which Logistic Regression
handles well; Random Forest is generally less suited to sparse text data.
Logistic Regression is also fast to train and highly interpretable, making it
a strong, standard baseline for text classification tasks like this.

Typos were not specifically handled, since AG News is professionally-written
news text (Reuters/AP), making misspellings rare. The 20k max-feature cap also
naturally filters out most rare/misspelled tokens.

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