from sklearn.model_selection import train_test_split
from random_forest import RandomForestRegressor

def train_model(df):
    # Split the data into features and target variable
    X = df[['UnitPrice', 'CustomerID', 'Country_Code', 'StockCodeInt']].values
    y = df['Quantity'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators=100)
    rf_regressor.fit(X_train, y_train)

    return rf_regressor, X_test, y_test
