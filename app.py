import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load trained model and dataset
model = joblib.load("virality_pipeline.joblib")
data = pd.read_csv("viral_posts.csv")
from sklearn.feature_extraction.text import TfidfVectorizer


# App Header
st.title("🔥 Virality Predictor AI")
st.write("Compare two social media posts and see which is more likely to go viral.")

# Input: Post A and Post B
col1, col2 = st.columns(2)
with col1:
    post_a = st.text_area("Post A Text", height=150)
with col2:
    post_b = st.text_area("Post B Text", height=150)

# Button to compare
if st.button("Predict Virality"):
    if not post_a.strip() or not post_b.strip():
        st.warning("Please enter text for both posts.")
    else:
        # Make dummy numeric features (replace with real logic if needed)
        def make_features(text):
            return {
                "text": text,
                "num_hashtags": text.count("#"),
                "num_emojis": sum(1 for c in text if c in "🔥🎉😂😍💯💥🛍️"),
                "post_length": len(text)
            }

        df_input = pd.DataFrame([make_features(post_a), make_features(post_b)])
        probs = model.predict_proba(df_input)[:, 1]

        st.subheader("Predicted Virality")
        col1.metric("Post A", f"{probs[0]*100:.2f}%")
        col2.metric("Post B", f"{probs[1]*100:.2f}%")

        # Confidence Messaging
        for i, p in enumerate(probs):
            level = "🔥 High" if p >= 0.85 else "⚠️ Moderate" if p >= 0.6 else "❌ Low"
            st.write(f"**Confidence for Post {'A' if i==0 else 'B'}:** {level}")

# Engagement Curve (bonus chart)
with st.expander("📊 Engagement Curve from Dataset"):
    if "engagement" in data.columns:
        # Re-score the data
        X_data = data[["text", "num_hashtags", "num_emojis", "post_length"]]
        data["prob"] = model.predict_proba(X_data)[:, 1]
        data = data.sort_values("prob", ascending=False)
        data["cum_eng"] = data["engagement"].cumsum()
        data["cum_pct_eng"] = data["cum_eng"] / data["engagement"].sum()
        data["cum_pct_posts"] = np.arange(len(data)) / len(data)

        fig, ax = plt.subplots()
        ax.plot(data["cum_pct_posts"], data["cum_pct_eng"])
        ax.set_title("Cumulative Engagement vs. Ranked Posts")
        ax.set_xlabel("Proportion of Posts")
        ax.set_ylabel("Cumulative Engagement")
        st.pyplot(fig)
    else:
        st.warning("Dataset does not include 'engagement' column.")
        