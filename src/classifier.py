"""
classifier.py
-------------
XGBoost 기반 음악 장르 분류 모듈.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


class GenreClassifier:
    """
    GTZAN CSV 데이터로 XGBoost 장르 분류 모델을 학습/추론합니다.

    Examples
    --------
    >>> clf = GenreClassifier()
    >>> clf.load_data('Data/features_3_sec.csv')
    >>> clf.train()
    >>> clf.evaluate()
    >>> clf.save('models/genre_clf.pkl')
    """

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: XGBClassifier | None = None
        self.scaler: MinMaxScaler | None = None
        self.le: LabelEncoder | None = None
        self.feature_cols: list[str] = []

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    def load_data(self, csv_path: str, test_size: float = 0.2):
        df = pd.read_csv(csv_path)

        X = df.drop(columns=["filename", "length", "label"])
        y_raw = df["label"]

        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y_raw)

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_cols = X.columns.tolist()

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X_scaled, y_enc,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_enc
        )
        print(f"Train: {self.X_train.shape} | Test: {self.X_test.shape}")
        print(f"장르: {list(self.le.classes_)}")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    def train(self):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.random_state,
        )
        self.model.fit(self.X_train, self.y_train, verbose=False)
        print("학습 완료 ✅")

    # ── 평가 ─────────────────────────────────────────────────────────────────
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("\n" + classification_report(
            self.y_test, preds,
            target_names=self.le.classes_
        ))
        return acc

    def confusion_matrix(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=self.le.classes_,
            yticklabels=self.le.classes_,
            cmap="Blues"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def feature_importance(self, top_n: int = 15):
        import matplotlib.pyplot as plt

        fi = pd.Series(self.model.feature_importances_, index=self.feature_cols)
        top = fi.nlargest(top_n).sort_values()

        plt.figure(figsize=(10, 5))
        top.plot(kind="barh", color="steelblue")
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    # ── 단일 예측 ─────────────────────────────────────────────────────────────
    def predict(self, feature_dict: dict) -> str:
        """
        특성 딕셔너리를 받아 장르명을 반환합니다.
        features.py의 extract_features() 결과를 그대로 전달하세요.
        """
        feat_vec = np.array([feature_dict.get(c, 0.0) for c in self.feature_cols]).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_vec)
        pred_idx = self.model.predict(feat_scaled)[0]
        return self.le.inverse_transform([pred_idx])[0]

    # ── 저장 / 로드 ──────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"모델 저장: {path}")

    @classmethod
    def load(cls, path: str) -> "GenreClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
