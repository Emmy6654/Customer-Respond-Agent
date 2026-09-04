import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class IntentAgent:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self.is_trained = False

    def train(self, csv_path, text_col="Ticket Description", label_col="Ticket Type"):
        df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
        X, y = df[text_col], df[label_col]
        can_stratify = y.value_counts().min() >= 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
        )
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print(f"[IntentAgent] Trained on {len(X_train)} tickets.")

    def predict(self, ticket_description):
        return self.pipeline.predict([ticket_description])[0]
