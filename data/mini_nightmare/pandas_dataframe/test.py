from pandas_dataframe_code import calculate_stats, load_data


def test_calculate_stats():
    df = load_data()
    median = calculate_stats(df)

    assert median == 14.4542
