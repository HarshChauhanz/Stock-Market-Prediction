import pandas as pd

def create_date_features(df, date_column='Date'):
    """
    Generates time-based features from a date column.
    """
    df = df.copy()
    # 'OrdinalDate' is crucial: it converts a date into a single continuous number (e.g., 737425)
    df['OrdinalDate'] = df[date_column].apply(lambda x: x.toordinal())
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['DayOfWeek'] = df[date_column].dt.dayofweek

    return df

def get_feature_columns():
    """Returns the list of features used for training."""
    return ['OrdinalDate', 'Year', 'Month', 'DayOfWeek']