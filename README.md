# ResolveAI — Smart Complaint Classifier

A complete classical-NLP web application that routes civic complaints to the correct department, predicts urgency, extracts important terms and retrieves the most textually similar known complaint.

## NLP techniques demonstrated

- Text normalization
- English stop-word removal
- TF-IDF weighting
- Unigrams and bigrams (`ngram_range=(1, 2)`)
- Sublinear term frequency
- Logistic Regression classification
- Multinomial Naive Bayes model comparison
- Separate multi-output pipelines for category and priority
- Prediction probabilities and top-3 classes
- TF-IDF keyword extraction
- Cosine-similarity document retrieval
- Classification report, accuracy and stratified train/test split

## Classes

Electricity, Water Supply, Roads & Transport, Sanitation, Public Safety, and Internet & Telecom.

## Files

```text
resolveai-nlp-vercel/
├── app.py
├── train_model.py
├── requirements.txt
├── vercel.json
├── README.md
├── LICENSE
├── data/complaints.csv
├── model/resolveai_nlp.pkl
├── static/app.js
├── static/style.css
└── templates/index.html
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The trained model is included. Run `python train_model.py` only to regenerate the dataset and model.

## Deploy on Vercel

1. Push every file and folder to a GitHub repository.
2. Import the repository in Vercel.
3. Use the detected **Flask** framework preset.
4. Keep the root directory as `./`; leave build/output commands blank.
5. Deploy. No environment variables are required.

## API

- `GET /api/health`
- `POST /api/analyse` with JSON: `{"complaint": "your complaint text"}`

## Dataset disclaimer

The included dataset is a deterministic, labelled synthetic dataset created by `train_model.py` for education and portfolio demonstration. It must not be represented as real municipal complaint data. Replace it with consented, representative data before production use.

