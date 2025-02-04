import pandas as pd


def load_data():
    print("Loading data...")

    # URL of the Titanic dataset (CSV format)
    url = "https://gist.githubusercontent.com/teamtom/1af7b484954b2d4b7e981ea3e7a27f24/raw/114fb69dce56b4462a9c3a417e7402330616ad4f/titanic_full.csv"

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
