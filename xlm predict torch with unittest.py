import numpy as np
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import logging
import unittest


logging.basicConfig(filename='prediction.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.manual_seed(42)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def download_data(symbol, start_date, end_date):
    try:
        logging.info("Fetching data...")
        data = yf.download(symbol, start=start_date, end=end_date)
        logging.info("Data fetched successfully.")
        print(data.head(10))
        print(data.columns)
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {str(e)}")
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            logging.info("Fetching data again...")
            data = yf.download(symbol, start=start_date, end=end_date)
            logging.info("Data fetched successfully.")
            print(data.head(10))
            print(data.columns)
        except Exception as e:
            logging.error(f"An error occurred while fetching data: {str(e)}")
            return None, None, None

    data = data.reset_index()
    data.columns = data.columns.str.strip()  # Remove leading and trailing whitespaces
    close_column = data.columns[
        (data.columns.str.contains('close', case=False)) & (data.columns.str.len() <= 6)
    ].tolist()
    if not close_column:
        logging.warning("No column with 'close' or 'adj close' found.")
        return None, None, None
    close_column = close_column[0]

    volume_column = data.columns.str.strip().tolist()
    volume_column = [col for col in volume_column if 'volume' in col.lower()]
    if not volume_column:
        logging.warning("No column with 'Volume' or similar name found.")
        return None, None, None
    volume_column = volume_column[0]

    return data, close_column, volume_column


def create_lagged_features(prices, volume_column, window_size=30):
    num_samples = len(prices) - window_size
    X = []
    y = []
    prices_df = pd.DataFrame(prices)  # Convert Series to DataFrame

    close_column = prices_df.columns[prices_df.columns.str.contains('close', case=False)].tolist()
    if not close_column:
        close_column = prices_df.columns[prices_df.columns.str.contains('adj close', case=False)].tolist()
    if not close_column:
        logging.warning("No column with 'Close' or 'Adj Close' found.")
        return None, None
    close_column = close_column[0]

    if volume_column is None or volume_column[0] not in prices_df.columns:
        logging.warning("Volume column not found or invalid. Skipping volume calculation.")
        for i in range(num_samples):
            window = prices_df[close_column][i:i + window_size]
            high = np.max(window)
            low = np.min(window)
            X.append(np.concatenate([window, [high, low]]))
            y.append(prices_df[close_column][i + window_size])
    else:
        volume_column = volume_column[0]
        for i in range(num_samples):
            window = prices_df[close_column][i:i + window_size]
            high = np.max(window)
            low = np.min(window)
            volume = np.array(prices_df[volume_column][i:i + window_size])
            X.append(np.concatenate([window, [high, low, volume]]))
            y.append(prices_df[close_column][i + window_size])

    X = np.array(X)
    y = np.array(y)
    return X, y


def calculate_rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test


def train_evaluate(args):
    X_train, y_train, X_test, y_test, learning_rate, max_depth = args
    model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds), model

def hyperparameter_tuning(X_train, y_train, X_test, y_test, learning_rates, max_depths):
    # Grid search or random search for hyperparameter tuning
    param_grid = {'learning_rate': learning_rates, 'max_depth': max_depths}
    model = xgb.XGBRegressor(objective='reg:squarederror')
    search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    search.fit(X_train, y_train)

    # Print MSE scores for each model
    mse_scores = -search.cv_results_['mean_test_score']
    for i, mse in enumerate(mse_scores):
        logging.info(f"Model {i + 1} MSE: {mse}")

    best_model = search.best_estimator_
    return best_model


def predict_next_days(model, last_window, num_days):
    preds = []
    highs = []
    lows = []
    window = list(last_window)
    for _ in range(num_days):
        pred = model.predict(np.array(window).reshape(1, -1))[0]
        preds.append(pred)
        high = np.max(window)
        low = np.min(window)
        highs.append(high)
        lows.append(low)
        window.pop(0)
        window.append(pred)
    return preds, highs, lows

def main():
    data, close_column, volume_column = download_data('XLM-USD', '2015-06-29', '2023-06-29')
    if data is None:
        return None
    prices = data[close_column]
    X, y = create_lagged_features(prices, volume_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    learning_rates = np.linspace(0.01, 0.2, 5)
    max_depths = np.arange(3, 8)

    best_model = hyperparameter_tuning(X_train, y_train, X_test, y_test, learning_rates, max_depths)

    # Now we use the best model to make a prediction for the next 20 days.
    last_window = X[-1]
    future_prices, future_highs, future_lows = predict_next_days(best_model, last_window, 20)

    # create a DataFrame
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 21)]

    predicted_prices = pd.DataFrame({
        'date': [d.strftime("%B %d, %Y") for d in future_dates],
        'predicted_price': future_prices,
        'predicted_high': future_highs,
        'predicted_low': future_lows
    })
    predicted_prices['predicted_price'] = predicted_prices['predicted_price'].astype(float)
    predicted_prices['predicted_high'] = predicted_prices['predicted_high'].astype(float)
    predicted_prices['predicted_low'] = predicted_prices['predicted_low'].astype(float)

    directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'Crypto')
    os.makedirs(directory, exist_ok=True)

    date_today = datetime.now().strftime("%m-%d-%y")
    file_path = os.path.join(directory, f"predicted_prices_torch_{date_today}.csv")
    predicted_prices.to_csv(file_path, index=False)

    print("File saved successfully...")

    return predicted_prices

if __name__ == '__main__':
    predicted_prices_torch = main()
    print(predicted_prices_torch)


