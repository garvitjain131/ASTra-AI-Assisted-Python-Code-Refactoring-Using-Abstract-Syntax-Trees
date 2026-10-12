import os
import sys
import math

PASS_MARK = 30 + 10


def average(marks):
    # Sum all the marks manually
    total = 0
    for m in marks:
        total += m
    avg = total / len(marks)
    return avg


def passed(mark):
    if mark >= PASS_MARK:
        return True
    else:
        return False


def grade(mark):
    if mark >= 75:
        letter = "A"
    else:
        letter = "B"
    return letter


def passing_students(names, marks):
    result = []
    for i in range(len(names)):
        if passed(marks[i]) == True:
            result.append(names[i])
    return result


def top_scores(marks):
    high = []
    for m in marks:
        if m > 80:
            high.append(m)
    return high


names = ["Asha", "Ben", "Chen", "Dev"]
marks = [91, 35, 78, 52]
print("Average:", average(marks))
print("Passed:", passing_students(names, marks))
print("Top:", top_scores(marks))
print("Grades:", [grade(m) for m in marks])
