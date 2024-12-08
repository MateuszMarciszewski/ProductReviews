from app_store_scraper import AppStore
import pandas as pd

# Initializes the AppStore object
app = AppStore(country="us", app_name="PocketGuard", app_id="949414211")

# Fetch reviews
app.review(how_many=2000)

# Convert reviews to a DataFrame
df = pd.DataFrame(app.reviews)

#check output
df.head()
# Save to CSV
df.to_csv('PocketGuard_app_reviews.csv', index=False)
