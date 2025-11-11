import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
from src.data_loader import load_and_clean_data
from src.features import create_date_features, get_feature_columns
from src.model_trainer import train_model, save_model, load_model

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
TARGET_COLUMN = 'Close'

def train_all_banks():

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in '{DATA_DIR}' directory.")
        print("Please make sure you put your 10 bank CSV files inside the 'data' folder.")
        return

    print(f"Found {len(csv_files)} datasets. Starting training process...")

    for file_path in csv_files:
        bank_name = os.path.basename(file_path).replace('.csv', '')
        print(f"\n--- Processing Bank: {bank_name} ---")

        try:
            df = load_and_clean_data(file_path)

            df_features = create_date_features(df)
            FEATURE_COLS = get_feature_columns()
            X = df_features[FEATURE_COLS]
            y = df_features[TARGET_COLUMN]

            model = train_model(X, y)

            model_filename = os.path.join(MODELS_DIR, f"{bank_name}_model.pkl")
            save_model(model, model_filename)
            print(f"SUCCESS: Finished processing {bank_name}.")
        except Exception as e:
            print(f"ERROR processing {bank_name}: {e}")

def predict_for_bank(bank_name, date_str):
    """
    Predicts future price for a SPECIFIC bank.
    """
    model_path = os.path.join(MODELS_DIR, f"{bank_name}_model.pkl")

    if not os.path.exists(model_path):
        print(f"Error: No trained model found for {bank_name}. Did you run training?")
        return None

    try:
        model = load_model(model_path)
        future_date = pd.to_datetime(date_str)
        input_df = pd.DataFrame({'Date': [future_date]})
        input_features = create_date_features(input_df)
        X_input = input_features[get_feature_columns()]
        prediction = model.predict(X_input)[0]
        return prediction
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

if __name__ == "__main__":
    print("Starting first-time setup...")
    train_all_banks()
    print("\nSetup Complete")

    test_bank = "SBI Dataset"
    test_date = "2025-03-15"

    print(f"\nRunning test prediction for {test_bank} on {test_date}...")
    price = predict_for_bank(test_bank, test_date)
    if price is not None:
        print(f"SUCCESS: Predicted price for {test_bank} on {test_date} is: {price:.2f}")
    else:
        print(f"Test prediction failed. Ensure '{test_bank}.csv' exists in 'data/' folder.")
