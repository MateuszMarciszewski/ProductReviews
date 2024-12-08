from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('PocketGuard_reviews_with_sentiment.csv')

# Filter for negative reviews
negative_reviews = df[df['sentiment'] == 'negative']['review']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
# Try 30, 40, 50?
tfidf_matrix = vectorizer.fit_transform(negative_reviews)

keywords = vectorizer.get_feature_names_out()

#need to screenshot the output.
print("Top Keywords from Negative Reviews:")
print(keywords)