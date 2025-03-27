from purr_code import cat_status


def test_cat_status():
    assert cat_status("Nono", 0) == "Nono: Purr"
    assert cat_status("Nono", 1) == "Nono: Purr"
    assert cat_status("Nono", 2) == "Nono: Purr"
    assert cat_status("Nono", 3) == "Nono: Meow!"
    assert cat_status("Nono", 4) == "Nono: Meow!"
    assert cat_status("Nono", 5) == "Nono: Meow!"
    assert cat_status("Nono", 6) == "Nono: MEOW!!!"
    assert cat_status("Nono", 7) == "Nono: Meow!"
    assert cat_status("Nono", 8) == "Nono: MEOW!!!"
    assert cat_status("Nono", 9) == "Nono: MEOW!!!"
