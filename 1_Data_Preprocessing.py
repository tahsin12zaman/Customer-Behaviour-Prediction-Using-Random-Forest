import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame

    Returns:
    DataFrame: DataFrame after handling missing values
    """
    # Remove rows with null or NaN values in the 'CustomerID' column
    df.dropna(subset=['CustomerID'], inplace=True)

    # Calculate mean for rest of the numeric columns
    numeric_cols = df.select_dtypes(include='number')
    df.fillna(numeric_cols.mean(), inplace=True)

    return df


def encode_categorical_variables(df):
    """
    Encode categorical variables in the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame

    Returns:
    DataFrame: DataFrame after encoding categorical variables
    """
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encoding the 'Country' column
    df['Country_Code'] = label_encoder.fit_transform(df['Country'])

    # Encoding the 'StockCode' column
    df['StockCodeInt'] = label_encoder.fit_transform(df['StockCode'])

    return df


def feature_scaling(numeric_df):
    """
    Perform feature scaling on the numeric DataFrame.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame

    Returns:
    DataFrame: Scaled numeric DataFrame
    """
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Perform feature scaling
    scaled_features = scaler.fit_transform(numeric_df)

    # Convert scaled features array to DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=numeric_df.columns)

    return scaled_features_df


def dimensionality_reduction_pca(numeric_df):
    """
    Perform dimensionality reduction on the numeric DataFrame using PCA.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame

    Returns:
    array: Principal components after PCA
    """
    # Initialize PCA
    pca = PCA(n_components=2)

    # Perform dimensionality reduction using PCA
    principal_components = pca.fit_transform(numeric_df)

    return principal_components


def dimensionality_reduction_tsne(numeric_df):
    """
    Perform dimensionality reduction on the numeric DataFrame using t-SNE.

    Parameters:
    numeric_df (DataFrame): Numeric DataFrame

    Returns:
    array: Components after t-SNE
    """
    # Initialize t-SNE
    tsne = TSNE(n_components=2)

    # Perform dimensionality reduction using t-SNE
    tsne_components = tsne.fit_transform(numeric_df)

    return tsne_components
