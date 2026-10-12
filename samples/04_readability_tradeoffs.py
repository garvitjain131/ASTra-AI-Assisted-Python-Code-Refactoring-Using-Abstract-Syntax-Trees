# Cases where the ML ranker should NOT auto-apply a valid refactoring.


def monthly_payment(principal, annual_rate, years):
    # A descriptive variable explaining a complex formula is worth keeping.
    monthly_installment = principal * (annual_rate / 12) / (1 - (1 + annual_rate / 12) ** (-years * 12))
    return monthly_installment


def matrix_pairs(rows, cols, limit):
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if r * c < limit:
                pairs.append((r, c))
    return pairs


def describe_temperature(celsius, humidity, wind_speed):
    if celsius > 30 and humidity > 70 or wind_speed < 5 and celsius > 35:
        label = "dangerously hot and humid conditions"
    else:
        label = "comfortable or moderately warm conditions"
    return label


SECONDS_PER_WEEK = 60 * 60 * 24 * 7
RATIO = 22 / 7
print(monthly_payment(1000, 0.05, 2) > 0, len(matrix_pairs(3, 3, 4)))
print(describe_temperature(36, 80, 2), SECONDS_PER_WEEK, RATIO)
