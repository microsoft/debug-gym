import pandas as pd


def load_data():
    # URL of the Titanic dataset (CSV format)
    url = "https://gist.githubusercontent.com/chisingh/d004edf19fffe92331e153a39466d38c/raw/titanic.csv"

    # Read the CSV file
    df = pd.read_csv(url)

    # Display the first few rows of the dataframe
    print(df.head())
    return df

def calculate_stats(df):
    # Calculate the median price paid
    median_price = df['Price'].median()
    return median_price
