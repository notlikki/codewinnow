def number_of_evens(items):
    answer = 0
    for item in items:
        if item % 2 == 0:
            answer += 1
    return answer
