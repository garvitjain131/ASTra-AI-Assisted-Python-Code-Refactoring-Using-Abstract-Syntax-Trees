"""Small inventory manager used to demo dead code and dictionary idioms."""
import json
from collections import OrderedDict

DAY = 60 * 60 * 24


def stock_level(inventory, item):
    if item in inventory:
        qty = inventory[item]
    else:
        qty = 0
    return qty


def needs_restock(inventory, item, minimum):
    level = stock_level(inventory, item)
    unit = "pcs"  # never used
    if not level >= minimum:
        return True
    return False


def apply_discount(price, code):
    if code == "SALE" or code == "PROMO" or code == "VIP":
        price = price * 0.9
    if price == None:
        return 0
    return price
    print("unreachable")


def describe(inventory):
    if len(inventory) == 0:
        return "empty"
    else:
        lines = {}
        for name in inventory:
            lines[name] = inventory[name] * 2
        return lines


def in_range(value, low, high):
    if low <= value and value <= high:
        if value != 0:
            return "ok"
    return "out of range"


inventory = {"apple": 5, "pear": 0, "plum": 12}
print(stock_level(inventory, "apple"), stock_level(inventory, "kiwi"))
print(needs_restock(inventory, "pear", 3))
print(apply_discount(100, "VIP"), apply_discount(100, "NONE"))
print(describe(inventory), describe({}))
print(in_range(5, 1, 10), in_range(0, -1, 1), in_range(50, 1, 10))
