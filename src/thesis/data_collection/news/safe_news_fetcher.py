import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

API_TOKEN = 'M3F3kC6r0Icz1sA1t70zwy4k8wc6UzpslRiRJpyr'
symbol = 'NVDA'
days_back = 1825
max_per_day = 25
output_file = f"{symbol}_news_safe.csv"

# Load existing data if file exists
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    fetched_dates = set(existing_df['date'].unique())
    all_articles = existing_df.to_dict('records')
    print(f"Resuming with {len(existing_df)} articles from {len(fetched_dates)} days.")
else:
    all_articles = []
    fetched_dates = set()

# Start from today and go back
end_date = datetime.utcnow()

try:
    for i in range(days_back):
        day = end_date - timedelta(days=i)
        date_str = day.strftime('%Y-%m-%d')
        if date_str in fetched_dates:
            continue  # Already fetched

        url = 'https://api.stockdata.org/v1/news/all'
        params = {
            'api_token': API_TOKEN,
            'symbols': symbol,
            'date_from': date_str,
            'date_to': date_str,
            'limit': max_per_day,
            'sort': 'asc'
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_data = response.json().get('data', [])
            for article in news_data:
                article['date'] = date_str
            all_articles.extend(news_data)

            # Save after each day
            pd.DataFrame(all_articles).to_csv(output_file, index=False)
            print(f"[{date_str}] Saved {len(news_data)} articles. Total so far: {len(all_articles)}")

        else:
            print(f"[{date_str}] Failed: {response.status_code}")
 
except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user. Saving current progress...")
    pd.DataFrame(all_articles).to_csv(output_file, index=False)
    print(f"✅ Saved {len(all_articles)} total articles to {output_file}")
