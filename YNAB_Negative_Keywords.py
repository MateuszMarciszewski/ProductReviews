from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('YNAB_reviews_with_sentiment.csv')

# Filter for negative reviews
negative_reviews = df[df['sentiment'] == 'negative']['review']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_matrix = vectorizer.fit_transform(negative_reviews)
keywords = vectorizer.get_feature_names_out()

print("Top Keywords from Negative Reviews:")
print(keywords)