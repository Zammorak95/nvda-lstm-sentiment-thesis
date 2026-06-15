from pytrends.request import TrendReq
import pandas as pd

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=0)

# Keywords to track
keywords = ["NVIDIA", "NVIDIA stock", "NVDA"]
pytrends.build_payload(keywords, timeframe='today 5-y', geo='US')

# -------------------------------------
# 1. Interest over time
# -------------------------------------
interest = pytrends.interest_over_time()
if 'isPartial' in interest.columns:
    interest = interest.drop(columns=['isPartial'])

interest.reset_index(inplace=True)
interest.to_csv("nvidia_interest_over_time.csv", index=False)
print("✅ Saved interest over time to nvidia_interest_over_time.csv")

# -------------------------------------
# 2. Related queries (for "NVIDIA" only)
# -------------------------------------
try:
    related_queries = pytrends.related_queries()
    if 'NVIDIA' in related_queries:
        rq = related_queries['NVIDIA']
        if rq:
            if rq.get('top') is not None:
                rq['top'].to_csv("nvidia_related_queries_top.csv", index=False)
                print("✅ Saved top related queries")
            if rq.get('rising') is not None:
                rq['rising'].to_csv("nvidia_related_queries_rising.csv", index=False)
                print("✅ Saved rising related queries")
        else:
            print("⚠️ No related queries found for 'NVIDIA'")
    else:
        print("⚠️ No related queries returned.")
except Exception as e:
    print(f"❌ Failed to fetch related queries: {e}")

# -------------------------------------
# 3. Related topics (for "NVIDIA" only)
# -------------------------------------
try:
    related_topics = pytrends.related_topics()
    if 'NVIDIA' in related_topics:
        rt = related_topics['NVIDIA']
        if rt:
            if rt.get('top') is not None:
                rt['top'].to_csv("nvidia_related_topics_top.csv", index=False)
                print("✅ Saved top related topics")
            if rt.get('rising') is not None:
                rt['rising'].to_csv("nvidia_related_topics_rising.csv", index=False)
                print("✅ Saved rising related topics")
        else:
            print("⚠️ No related topics found for 'NVIDIA'")
    else:
        print("⚠️ No related topics returned.")
except Exception as e:
    print(f"❌ Failed to fetch related topics: {e}")
