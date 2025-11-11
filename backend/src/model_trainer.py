from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(X, y):
    """Trains the HistGradientBoostingRegressor."""
    print("Training model...")
    # Random state ensures reproducible results
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    print("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Prints basic evaluation metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error on Test Set: {mae:.2f} INR")

def save_model(model, filename='models/stock_price_model.pkl'):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='models/stock_price_model.pkl'):
    """Loads a trained model from disk."""
    return joblib.load(filename)