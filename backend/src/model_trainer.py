from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model(X, y):
    """
    Trains a Random Forest Regressor model using the given features (X) and target (y).
    Returns the trained model.
    """
    print("Training Random Forest Regressor model...")

    model = RandomForestRegressor(
        n_estimators=100,   
    )

   
    model.fit(X, y)

    print("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using Mean Absolute Error and R2 Score.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return mae, r2


def save_model(model, filename):
    """
    Saves the trained model as a .pkl file.
    """
    joblib.dump(model, filename)
    print(f"Model saved successfully as {filename}")


def load_model(filename):
    """
    Loads a previously saved model from a .pkl file.
    """
    model = joblib.load(filename)
    print(f"Model loaded successfully from {filename}")
    return model
