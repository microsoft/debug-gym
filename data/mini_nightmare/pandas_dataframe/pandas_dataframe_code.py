import pandas as pd


def load_data():
    # URL of the Titanic dataset (CSV format)
    url = "https://gist.githubusercontent.com/chisingh/d004edf19fffe92331e153a39466d38c/raw/titanic.csv"

    # Read the CSV file
    df = pd.read_csv(url)

    return df

def calculate_stats(df):
    # Calculate the median fare paid
    median = df['Price'].median()
    print("\nMedian fare paid:\n", median)
    return median
