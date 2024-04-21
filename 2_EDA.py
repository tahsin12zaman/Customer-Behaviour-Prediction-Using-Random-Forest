import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans

def summary_statistics(numeric_df):
    """
    Calculate summary statistics for numeric columns.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame

    Returns:
    DataFrame: Summary statistics DataFrame
    """
    return numeric_df.describe()


def univariate_analysis(numeric_df):
    """
    Perform univariate analysis on numeric columns.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame
    """
    numeric_df.hist(figsize=(12, 8))
    plt.show()


def bivariate_analysis(numeric_df, x_column, y_column):
    """
    Perform bivariate analysis on numeric columns.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame
    x_column (str): Name of the x-axis column
    y_column (str): Name of the y-axis column
    """
    sns.scatterplot(x=x_column, y=y_column, data=numeric_df)
    plt.show()


def correlation_heatmap(numeric_cols):
    """
    Generate a correlation heatmap for numeric columns.

    Parameters:
    numeric_cols (DataFrame): Numeric DataFrame
    """
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
    plt.show()


def create_histogram(df, column):
    """
    Create a histogram for a specific numeric column.

    Parameters:
    df (DataFrame): Input DataFrame
    column (str): Name of the column
    """
    sns.histplot(df[column])
    plt.title(f'Distribution of {column}')
    plt.show()


def create_countplot(df, column):
    """
    Create a count plot for a specific categorical column.

    Parameters:
    df (DataFrame): Input DataFrame
    column (str): Name of the column
    """
    sns.countplot(x=column, data=df)
    plt.title(f'Count of {column}')
    plt.xticks(rotation=45)
    plt.show()


def temporal_analysis(df, datetime_column):
    """
    Perform temporal analysis on a datetime column.

    Parameters:
    df (DataFrame): Input DataFrame
    datetime_column (str): Name of the datetime column
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column], format="%d-%m-%Y %H:%M")
    df['InvoiceMonth'] = df[datetime_column].dt.month

    sns.countplot(x='InvoiceMonth', data=df)
    plt.title('Number of Invoices by Month')
    plt.show()


def cluster_analysis(df, scaled_features_df, n_clusters=3):
    """
    Perform cluster analysis using K-means clustering.

    Parameters:
    df (DataFrame): Input DataFrame
    scaled_features_df (DataFrame): Scaled features DataFrame
    n_clusters (int): Number of clusters
    """


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_features_df)
    df['Cluster'] = kmeans.labels_

    # Plot clusters based on PCA
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df)
    plt.title('Clusters based on PCA')
    plt.show()
