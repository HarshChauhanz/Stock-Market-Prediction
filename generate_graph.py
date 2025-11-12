import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.data_loader import load_and_clean_data
from src.features import create_date_features, get_feature_columns
from src.model_trainer import load_model


MODEL_PATH = 'models/SBI Dataset_model.pkl'
DATA_PATH = 'data/SBI Dataset.csv' 

def generate_prediction_graph(target_date_str):
    """
    Generates and saves a graph showing the predicted trend around a target date.
    """
    
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Run training first.")
        return

   
    target_date = pd.to_datetime(target_date_str)
    start_date = target_date - dt.timedelta(days=30)
    end_date = target_date + dt.timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

   
    plot_df = pd.DataFrame({'Date': date_range})
    plot_features = create_date_features(plot_df)
    X_plot = plot_features[get_feature_columns()]

    
    plot_df['Predicted_Close'] = model.predict(X_plot)

    target_price = plot_df[plot_df['Date'] == target_date]['Predicted_Close'].values[0]

    
    plt.figure(figsize=(10, 6))
    
    plt.plot(plot_df['Date'], plot_df['Predicted_Close'], label='30-Day Predicted Trend', color='blue', linestyle='--', linewidth=2)
    
    plt.scatter([target_date], [target_price], color='red', s=150, zorder=5, label=f'Forecast for {target_date.date()}: {target_price:.2f}')

    
    plt.title(f'Predicted Stock Price Trend around {target_date.date()}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted Closing Price (INR)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

   
    output_filename = f'prediction_graph_{target_date.date()}.png'
    plt.savefig(output_filename)
    print(f"Graph saved as: {output_filename}")

if __name__ == '__main__':
  
    generate_prediction_graph("2025-03-15")