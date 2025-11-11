import pandas as pd

def load_and_clean_data(filepath):
    """
    Loads the stock dataset and performs initial cleaning.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert 'Date' column to datetime objects
    # dayfirst=True is important for DD-MM-YYYY formats common in Indian datasets
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Drop rows where essential data is missing
    df = df.dropna(subset=['Date', 'Close'])

    # Sort by date to ensure correct chronological order
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df