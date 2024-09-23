import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.logging import logger

def find_missing_values(df):
    """
    Finds missing values and returns a summary.

    Args:
        df: The DataFrame to check for missing values.

    Returns:
        A summary of missing values, including the number of missing values per column.
    """

    logger.info("Finding missing values in DataFrame...")
    null_counts = df.isnull().sum()
    missing_value = null_counts
    percent_of_missing_value = 100 * null_counts / len(df)
    data_type = df.dtypes

    missing_data_summary = pd.concat([missing_value, percent_of_missing_value, data_type], axis=1)
    missing_data_summary_table = missing_data_summary.rename(columns={0: "Missing values", 1: "Percent of Total Values", 2: "DataType"})
    missing_data_summary_table = missing_data_summary_table[missing_data_summary_table.iloc[:, 1] != 0].sort_values('Percent of Total Values', ascending=False).round(1)

    print(f"From {df.shape[1]} columns selected, there are {missing_data_summary_table.shape[0]} columns with missing values.")

    return missing_data_summary_table


def get_outlier_summary(data,column_names):
    """
    Calculates outlier summary statistics for a DataFrame.

    Args:
        data : Input DataFrame.

    Returns:
        Outlier summary DataFrame.
    """

    logger.info("Calculating outlier summary for numerical columns...")
    outlier_summary = pd.DataFrame(columns=['Variable', 'Number of Outliers'])
    # data = data.select_dtypes(include='number')

    for column_name in column_names:
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]

        outlier_summary = pd.concat(
            [outlier_summary, pd.DataFrame({'Variable': [column_name], 'Number of Outliers': [outliers.shape[0]]})],
            ignore_index=True
        )
    non_zero_count = (outlier_summary['Number of Outliers'] > 0).sum()
    print(f"From {data.shape[1]} selected numerical columns, there are {non_zero_count} columns with outlier values.")

    return outlier_summary


def replace_missing_values(data):
    """
    Replaces missing values in a DataFrame with the mean for numeric columns and the mode for categorical columns.

    Args:
        data: The input DataFrame.

    Returns:
        The DataFrame with missing values replaced.
    """

    logger.info("Replacing missing values...")
    numeric_columns = data.select_dtypes(include='number').columns
    categorical_columns = data.select_dtypes(include='object').columns

    # Replace missing values in numeric columns with the mean
    for column in numeric_columns:
        column_mean = data[column].max()
        logger.info(f"Replacing missing values in column '{column}' with mean: {column_mean}")
        data[column] = data[column].fillna(column_mean)

    # Replace missing values in categorical columns with the mode
    for column in categorical_columns:
        column_mode = data[column].mode().iloc[0]
        logger.info(f"Replacing missing values in column '{column}' with mode: {column_mode}")
        data[column] = data[column].fillna(column_mode)

    return data


def remove_outliers_winsorization(data,column_names):
    """
    Removes outliers from specified columns of a DataFrame using winsorization.

    Args:
        data: The input DataFrame.
        column_names (list): A list of column names to process.

    Returns:
        The DataFrame with outliers removed.
    """

    logger.info("Removing outliers using winsorization...")
    for column_name in column_names:
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        logger.info(f"Winsorizing column '{column_name}' with lower bound: {lower_bound} and upper bound: {upper_bound}")
        data[column_name] = data[column_name].clip(lower_bound, upper_bound)

    return data

def boxPlotForDetectOutliers(train_data,column_names):
    logger.info("Creating box plots for outlier detection...")
    for column in column_names:
        sns.boxplot(data=train_data[column])
        plt.title(f"Box Plot of {column}")
        plt.show()
