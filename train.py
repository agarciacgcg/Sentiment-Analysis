from data_loader import load_data
from features import build_preprocessor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import numpy as np

# 1. Load & split data
df = load_data("viral_posts.csv")
X = df[["text", "num_hashtags", "num_emojis", "post_length"]]
y = (df["viral_label"] == "Viral").astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Handle class imbalance
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# 3. Build preprocessor
prep = build_preprocessor()

# 4. Hyperparameter search - NO early stopping here!
search_pipeline = Pipeline([
    ("prep", prep),
    ("clf", XGBClassifier(
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=scale_pos_weight  # imbalance fix
    ))
])

param_dist = {
    "clf__n_estimators":  [100, 300, 500],
    "clf__max_depth":     [4, 6, 8],
    "clf__learning_rate": [0.01, 0.1, 0.2],
    "clf__subsample":     [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    error_score="raise"
)
search.fit(X_train, y_train)

# 5. Train final model WITH early stopping
best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

# Preprocess manually for final model
X_train_t = prep.fit_transform(X_train)
X_val_t = prep.transform(X_val)

final_clf = XGBClassifier(
    eval_metric="auc",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20,
    **best_params
)

final_clf.fit(
    X_train_t, y_train,
    eval_set=[(X_val_t, y_val)],
    verbose=False
)

# 6. Wrap everything into a pipeline
final_pipeline = Pipeline([
    ("prep", prep),
    ("clf", final_clf)
])

# 7. Evaluate performance
y_pred = final_pipeline.predict(X_val)
y_proba = final_pipeline.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_proba))

# 8. Save model
joblib.dump(final_pipeline, "virality_pipeline.joblib")