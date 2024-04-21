import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score

def calculate_performance_metrics(model, X_test, y_test):
    # Make predictions on the testing set
    predictions = model.predict(X_test)

    # Evaluate the model using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Evaluate the model using Root Mean Squared Error (RMSE)
    rmse = mse ** 0.5

    # Evaluate the model using R-squared (R2)
    r2 = r2_score(y_test, predictions)

    # Evaluate the model using Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, predictions)

    # Evaluate the model using Median Absolute Error
    medae = median_absolute_error(y_test, predictions)

    return mse, rmse, r2, mape, medae

def visualize_performance_metrics(metrics_values):
    metrics = ['MSE', 'RMSE', 'R2', 'MAPE', 'Median AE']
    values = metrics_values

    # Create a bar plot for performance metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color='skyblue')
    plt.xlabel('Performance Metric')
    plt.ylabel('Value')
    plt.title('Performance Metrics of Random Forest Regression Model')
    plt.xticks(rotation=45)
    plt.show()

def visualize_actual_vs_predicted(y_test, predictions):
    # Create a scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values of Quantity')
    plt.show()
