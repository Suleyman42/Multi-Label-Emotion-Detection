import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("data/processed/cleaned_track-a.csv")

# Take the first 5 samples (or use df.sample(5) for random ones)
test_df = df[['text']].head(5)

# Save to test.csv
test_df.to_csv("data/test/test.csv", index=False)

print("✅ test.csv created with 5 rows from cleaned_track-a.csv")
