from pathlib import Path
import re

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "model" / "resolveai_nlp.pkl")
category_model = bundle["category_model"]
priority_model = bundle["priority_model"]
known_texts = bundle["texts"]
known_categories = bundle["categories"]

app = Flask(__name__)


def normalize(text):
    text = re.sub(r"[^a-zA-Z0-9\s'-]", " ", text.lower())
    return " ".join(text.split())


def top_keywords(text, limit=6):
    vectorizer = category_model.named_steps["tfidf"]
    row = vectorizer.transform([text]).toarray()[0]
    names = vectorizer.get_feature_names_out()
    indices = np.argsort(row)[::-1]
    return [names[index] for index in indices if row[index] > 0][:limit]


def similar_complaint(text):
    vectorizer = category_model.named_steps["tfidf"]
    matrix = vectorizer.transform(known_texts)
    query = vectorizer.transform([text])
    scores = cosine_similarity(query, matrix)[0]
    index = int(np.argmax(scores))
    return {
        "text": known_texts[index].capitalize(),
        "category": known_categories[index],
        "similarity": round(float(scores[index]) * 100, 1),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": bundle["selected_model"], "vectorizer": "TF-IDF word n-grams"})


@app.post("/api/analyse")
def analyse():
    payload = request.get_json(silent=True) or {}
    text = normalize(str(payload.get("complaint", "")))
    if len(text) < 15:
        return jsonify({"error": "Please enter a complaint of at least 15 characters."}), 400
    if len(text) > 2500:
        return jsonify({"error": "Please keep the complaint under 2,500 characters."}), 400

    probabilities = category_model.predict_proba([text])[0]
    classes = category_model.classes_
    ranking = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    priority_probabilities = priority_model.predict_proba([text])[0]
    priority_classes = priority_model.classes_
    urgent_index = list(priority_classes).index("Urgent")

    return jsonify({
        "category": ranking[0][0],
        "confidence": round(float(ranking[0][1]) * 100, 1),
        "priority": priority_model.predict([text])[0],
        "urgency_score": round(float(priority_probabilities[urgent_index]) * 100, 1),
        "top_categories": [{"name": name, "score": round(float(score) * 100, 1)} for name, score in ranking[:3]],
        "keywords": top_keywords(text),
        "similar": similar_complaint(text),
        "cleaned_text": text,
    })


if __name__ == "__main__":
    app.run(debug=True)

