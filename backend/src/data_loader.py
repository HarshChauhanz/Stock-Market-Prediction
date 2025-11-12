import pandas as pd

def load_and_clean_data(filepath):
    """
    Loads the stock dataset and performs initial cleaning.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

   
    df = df.dropna(subset=['Date', 'Close'])

    
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df