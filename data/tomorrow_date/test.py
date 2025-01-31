from main import TomorrowDate


def test_tomorrow_date():

    td = TomorrowDate()
    # try some edge cases supported_formats = ['%Y-%m-%d', '%m-%d-%Y', '%m.%d.%Y', '%m %d %Y', '%m %d %y']
    test_cases = ["2025-12-31", "12-31-2025", "12.31.2025", "12 31 2025", "12 31 25", 
                  "2025-02-28", "02-28-2025", "02.28.2025", "02 28 2025", "02 28 25", 
                  "2025-01-31", "01-31-2025", "01.31.2025", "01 31 2025", "01 31 25", 
                  "2025-03-31", "03-31-2025", "03.31.2025", "03 31 2025", "03 31 25",
                  "2025-02-29", "02-29-2025", "02.29.2025", "02 29 2025", "02 29 25",
                  "2024-02-29", "02-29-2024", "02.29.2024", "02 29 2024", "02 29 24",
                  "2025-00-24", "00-24-2025", "00.24.2025", "00 24 2025", "00 24 25"]

    expected = ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01", 
                "2025-03-01", "2025-03-01", "2025-03-01", "2025-03-01", "2025-03-01",
                "2025-02-01", "2025-02-01", "2025-02-01", "2025-02-01", "2025-02-01",
                "2025-04-01", "2025-04-01", "2025-04-01", "2025-04-01", "2025-04-01",
                "None", "None", "None", "None", "None",
                "2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01",
                "None", "None", "None", "None", "None"]





    for i, test_case in enumerate(test_cases):
        assert str(td.tomorrow(test_case)) == expected[i]
