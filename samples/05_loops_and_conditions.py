def count_vowels(text):
    count = 0
    for ch in text:
        if ch in "aeiou":
            count += 1
    return count


def word_lengths(words):
    lengths = {}
    for w in words:
        lengths[w] = len(w)
    return lengths


def unique_initials(words):
    initials = set()
    for w in words:
        initials.add(w[0])
    return initials


def print_numbered(items):
    for i in range(len(items)):
        print(i + 1, items[i])


def classify(n):
    if n < 0:
        raise ValueError("negative")
    else:
        kind = "even" if n % 2 == 0 else "odd"
        return kind


def is_weekend(day):
    if not day in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        return True
    else:
        return False


def countdown(n):
    while n > 0:
        n = n - 1
        pass
    return n


words = ["apple", "banana", "cherry"]
print(count_vowels("refactoring"), word_lengths(words), sorted(unique_initials(words)))
print_numbered(words)
print(classify(4), is_weekend("Sun"), countdown(5))
