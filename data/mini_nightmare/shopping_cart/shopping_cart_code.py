class ShoppingCart:
    def __init__(self):
        self.items = {}
        self.total = 0.0
        self.discounts = {}
        self.item_prices = {}

    def add_item(self, item, price, quantity=1):
        if item in self.items:
            self.items[item] += quantity
        else:
            self.items[item] = quantity
            self.item_prices[item] = price

        self.total += price * quantity

    def apply_discount(self, item, discount_percentage):
        self.discounts[item] = discount_percentage
        if item in self.items:
            price = self.item_prices[item]
            self.total -= (price * self.items[item] * discount_percentage / 100)

    def get_total(self):
        return round(self.total, 2)
