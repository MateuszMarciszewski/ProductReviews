import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load reviews from the CSV
df = pd.read_csv('PocketGuard_app_reviews.csv')

# Initializes VADER (Try HuggingFace if this doesn't work)
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each review
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review'].apply(analyze_sentiment)

# Save the results
df.to_csv('PocketGuard_reviews_with_sentiment.csv', index=False)
