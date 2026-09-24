"""Generate the demo dataset, compare NLP classifiers, and save the best pipelines."""

import csv
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
SEED = 42

CATEGORIES = {
    "Electricity": ["power cut", "street light", "electricity outage", "sparking wire", "low voltage", "electric pole", "meter fault", "transformer noise"],
    "Water Supply": ["no water", "dirty water", "pipeline leak", "low water pressure", "sewage mixed water", "broken water pipe", "irregular supply", "water wastage"],
    "Roads & Transport": ["pothole", "broken road", "traffic signal", "damaged footpath", "unsafe crossing", "bus delay", "waterlogged road", "missing road sign"],
    "Sanitation": ["garbage pile", "missed waste collection", "blocked drain", "bad smell", "overflowing dustbin", "mosquito breeding", "open dumping", "unclean public toilet"],
    "Public Safety": ["open manhole", "fallen tree", "unsafe construction", "stray animal", "dark street", "damaged railing", "fire hazard", "exposed cable"],
    "Internet & Telecom": ["internet down", "slow broadband", "mobile network", "frequent disconnection", "damaged cable", "no signal", "router outage", "poor coverage"],
}

LOCATIONS = ["near my house", "outside the school", "in our lane", "at the main market", "near the bus stand", "in sector 12", "beside the park", "at the crossing"]
TEMPLATES = [
    "There is {issue} {location} since {duration}.",
    "Please resolve the {issue} {location}; it has continued for {duration}.",
    "I want to report {issue} {location}. The problem started {duration} ago.",
    "Residents are affected by {issue} {location} for {duration}.",
    "The local authority has not fixed the {issue} {location} for {duration}.",
]
DURATIONS = ["two hours", "one day", "three days", "a week", "several weeks"]
URGENT_MARKERS = ["dangerous", "emergency", "accident may happen", "children are at risk", "immediate help needed", "severe", "people may get hurt"]
NORMAL_MARKERS = ["please check", "kindly resolve", "causing inconvenience", "needs attention", "please inspect", "requesting repair"]


def clean_text(text):
    return " ".join(text.lower().strip().split())


def generate_dataset(samples_per_category=110):
    random.seed(SEED)
    rows = []
    for category, issues in CATEGORIES.items():
        for index in range(samples_per_category):
            urgent = index % 3 == 0
            text = random.choice(TEMPLATES).format(
                issue=random.choice(issues), location=random.choice(LOCATIONS), duration=random.choice(DURATIONS)
            )
            marker = random.choice(URGENT_MARKERS if urgent else NORMAL_MARKERS)
            if random.random() > .5:
                text = f"{marker.capitalize()}. {text}"
            else:
                text = f"{text} It is {marker}."
            rows.append((clean_text(text), category, "Urgent" if urgent else "Normal"))
    random.shuffle(rows)
    return rows


def pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=.96, sublinear_tf=True, max_features=7000)),
        ("classifier", classifier),
    ])


def main():
    rows = generate_dataset()
    texts = [row[0] for row in rows]
    categories = [row[1] for row in rows]
    priorities = [row[2] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(texts, categories, test_size=.22, random_state=SEED, stratify=categories)

    candidates = {
        "Multinomial Naive Bayes": pipeline(MultinomialNB(alpha=.4)),
        "Logistic Regression": pipeline(LogisticRegression(max_iter=1600, C=5, random_state=SEED)),
    }
    scores = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        scores[name] = accuracy_score(y_test, candidate.predict(x_test))
    best_name = max(scores, key=scores.get)
    category_model = candidates[best_name]

    p_train, p_test, py_train, py_test = train_test_split(texts, priorities, test_size=.22, random_state=SEED, stratify=priorities)
    priority_model = pipeline(LogisticRegression(max_iter=1200, C=4, class_weight="balanced", random_state=SEED))
    priority_model.fit(p_train, py_train)
    priority_accuracy = accuracy_score(py_test, priority_model.predict(p_test))

    ROOT.joinpath("data").mkdir(exist_ok=True)
    with ROOT.joinpath("data", "complaints.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["complaint_text", "category", "priority"])
        writer.writerows(rows)

    ROOT.joinpath("model").mkdir(exist_ok=True)
    joblib.dump({
        "category_model": category_model,
        "priority_model": priority_model,
        "texts": texts,
        "categories": categories,
        "comparison": {name: round(float(score), 4) for name, score in scores.items()},
        "selected_model": best_name,
        "category_accuracy": round(float(scores[best_name]), 4),
        "priority_accuracy": round(float(priority_accuracy), 4),
        "report": classification_report(y_test, category_model.predict(x_test), output_dict=True),
    }, ROOT / "model" / "resolveai_nlp.pkl", compress=3)

    print("Category comparison:", scores)
    print("Selected:", best_name)
    print("Priority accuracy:", round(priority_accuracy, 4))


if __name__ == "__main__":
    main()

