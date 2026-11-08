def summarize_numbers(values):
    total = 0
    even_count = 0
    positive_count = 0
    for value in values:
        total += value
        if value % 2 == 0:
            even_count += 1
        if value > 0:
            positive_count += 1
    average = total / len(values) if values else 0
    return total, average, even_count, positive_count
