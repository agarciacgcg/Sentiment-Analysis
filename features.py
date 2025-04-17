from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    text_pipe = ("tfidf", TfidfVectorizer(ngram_range=(1,2),
                                          stop_words="english",
                                          max_features=5000), "text")
    nums = ("nums", "passthrough", ["num_hashtags","num_emojis","post_length"])
    return ColumnTransformer([text_pipe, nums])