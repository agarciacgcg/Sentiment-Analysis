from data_loader import load_data
from features import build_preprocessor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

df = load_data("virality_predictor/Realistic_Viral_Post_Dataset.csv")
X = df[["text", "num_hashtags", "num_emojis", "post_length"]]
y = (df["viral_label"] == "Viral").astype(int)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

prep = build_preprocessor()
model = Pipeline([("prep", prep),
                  ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))])
model.fit(X_train, y_train)

joblib.dump(model, "virality_model.joblib")
