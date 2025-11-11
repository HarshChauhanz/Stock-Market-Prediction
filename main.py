import pandas as pd
import sys
import os
# Add the current folder to Python's path so it can find 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import os
import glob
from src.data_loader import load_and_clean_data
from src.features import create_date_features, get_feature_columns
from src.model_trainer import train_model, save_model, load_model

# --- CONFIGURATION ---
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
TARGET_COLUMN = 'Close'

def train_all_banks():
    """
    Loops through all CSV files in the data directory and trains a model for each.
    """
    # Get a list of all CSV files in the data folder
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in '{DATA_DIR}' directory.")
        print("Please make sure you put your 10 bank CSV files inside the 'data' folder.")
        return

    print(f"Found {len(csv_files)} datasets. Starting training process...")

    for file_path in csv_files:
        # Extract bank name from filename (e.g., 'data/HDFC.csv' -> 'HDFC')
        bank_name = os.path.basename(file_path).replace('.csv', '')
        print(f"\n--- Processing Bank: {bank_name} ---")

        try:
            # 1. Load Data
            df = load_and_clean_data(file_path)

            # 2. Feature Engineering
            df_features = create_date_features(df)
            FEATURE_COLS = get_feature_columns()
            X = df_features[FEATURE_COLS]
            y = df_features[TARGET_COLUMN]

            # 3. Train Model (using all data for final deployment)
            model = train_model(X, y)

            # 4. Save Model uniquely for this bank
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
    # --- FIRST TIME RUN SECTION ---
    print("Starting first-time setup...")
    train_all_banks()
    print("\n--- Setup Complete ---")
    print("You can now comment out 'train_all_banks()' and use the prediction section below.")

    # --- TEST PREDICTION SECTION ---
    # This will only work if you have a file named 'SBI Dataset.csv' in your data folder.
    # Change 'SBI Dataset' to match one of your actual filenames if needed.
    test_bank = "SBI Dataset"
    test_date = "2025-03-15"

    print(f"\nRunning test prediction for {test_bank} on {test_date}...")
    price = predict_for_bank(test_bank, test_date)
    if price is not None:
        print(f"SUCCESS: Predicted price for {test_bank} on {test_date} is: {price:.2f}")
    else:
        print(f"Test prediction failed. Ensure '{test_bank}.csv' exists in 'data/' folder.")
