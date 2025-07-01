import pandas as pd
from cleantext import clean

# Emotion label columns
EMOTION_COLS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df_raw)} rows")

    # Convert emotion labels to integers and fill missing
    df_raw[EMOTION_COLS] = (
        df_raw[EMOTION_COLS]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .astype(int)
    )
   

    # Remove rows with no emotions
    zero_before = (df_raw[EMOTION_COLS].sum(axis=1) == 0).sum()
    print(f"Rows with no emotions (before): {zero_before}")

    df = df_raw[df_raw[EMOTION_COLS].sum(axis=1) > 0].reset_index(drop=True)

    zero_after = (df[EMOTION_COLS].sum(axis=1) == 0).sum()
    print(f"Rows with no emotions (after): {zero_after}")
    print(f"Dataset size: before={len(df_raw)}, after={len(df)}")

    # Clean text
    df["text"] = df["text"].astype(str).apply(lambda t: clean(
        t,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False
    ))

    return df
def save_cleaned_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")