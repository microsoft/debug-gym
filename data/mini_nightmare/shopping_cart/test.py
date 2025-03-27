from shopping_cart_code import ShoppingCart


def test_shopping_cart():

    cart = ShoppingCart()
    cart.add_item("apple", 1.0, 2)  # 2 apples at $1 each
    cart.add_item("banana", 0.5, 3)  # 3 bananas at $0.5 each
    cart.add_item("orange", 0.75, 5)  # 5 orange at $0.75
    assert cart.get_total() == 7.25

    cart.apply_discount("apple", 20)  # 20% discount on apples
    cart.apply_discount("orange", 10)  # 10% discount on oranges
    cart.add_item("apple", 1.0, 3)  # Add more apples
    cart.add_item("orange", 0.75, 2)  # Add more oranges
    assert cart.get_total() == 10.23
