from code import calculate_stats, load_data


def test_calculate_stats():
    df = load_data()
    median_price = calculate_stats(df)

    assert median_price == 14.4542
