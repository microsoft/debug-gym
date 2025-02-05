import pandas as pd


def load_data():
    print("Loading data...")

    # URL of the Titanic dataset (CSV format)
    url = "https://gist.githubusercontent.com/chisingh/d004edf19fffe92331e153a39466d38c/raw/titanic.csv"

    # Read the CSV file
    df = pd.read_csv(url)

    # Display the first few rows of the dataframe
    print(df.head())
    return df

def calculate_stats(df):
    print("\nCalculating statistics...\n")

    # Calculate the median price paid
    median_price = df['Price'].median()
    print("\nMedian price paid:\n", median_price)
