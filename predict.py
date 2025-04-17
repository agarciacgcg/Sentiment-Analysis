import sys
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("virality_model.joblib")

# Function to extract features from raw text
def make_features(text):
    return {
        "text": text,
        "num_hashtags": text.count("#"),
        "num_emojis": sum(1 for c in text if c in "🔥🎉😂😍💯💥🛍️"),
        "post_length": len(text)
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py 'Post A text' 'Post B text'")
        sys.exit(1)

    text_a = sys.argv[1]
    text_b = sys.argv[2]

    df = pd.DataFrame([make_features(text_a), make_features(text_b)])
    probs = model.predict_proba(df)[:, 1]  # Get probability of 'Viral' class

    print("\nVirality Prediction:")
    print(f"Post A → {probs[0]*100:.2f}% chance of going viral")
    print(f"Post B → {probs[1]*100:.2f}% chance of going viral")

if __name__ == "__main__":
    main()