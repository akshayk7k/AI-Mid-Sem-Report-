import random

def generate_instance(n, k, m):
    vars = [chr(i + 65) for i in range(n)]
    problem = "(("
    clause = []

    for i in range(k * m):
        x = random.choice(vars)
        vars.remove(x)
        clause.append(x)

        if i % k == k - 1:
            vars.extend(clause)
            clause.clear()

        if random.random() < 0.5:
            problem += "~"
        problem += x

        if i % k == k - 1 and i != (k * m - 1):
            problem += ") and ("
        elif i != (k * m - 1):
            problem += " or "

    problem += "))"
    return problem

for i in range(10):
    print(f"Problem {i + 1}: {generate_instance(12, 3, 4)}")
