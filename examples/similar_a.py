def sum_positive_numbers(numbers):
    total = 0
    for number in numbers:
        if number > 0:
            total += number
    return total