import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Load the dataset
    df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
    print('Initial Data Info:')
    print(df.info())
    print('\nData Shape:', df.shape)
    print('\nMissing Values:\n', df.isnull().sum())

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df = df.sort_values(by=['id', 'date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Define target columns (pollutants)
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    # Drop rows with missing target values
    df = df.dropna(subset=pollutants)

    # Fill missing values in features with median
    feature_cols = ['NH4', 'BSK5', 'Suspended', 'year', 'month']
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols]
    y = df[pollutants]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print('\nModel Evaluation:')
    for i, pollutant in enumerate(pollutants):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f'{pollutant}: MSE={mse:.4f}, R2={r2:.4f}')

    # Show a few predictions
    print('\nSample Predictions:')
    print(pd.DataFrame(y_pred, columns=pollutants).head())

if __name__ == '__main__':
    main() 