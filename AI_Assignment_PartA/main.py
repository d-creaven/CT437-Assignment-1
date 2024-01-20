import random


def generate_solution():
    solution = '0b'
    for i in range(0, 30):
        solution += str(random.choice([0, 1]))
    return solution


def generate_initial_population():
    solutions = []
    for i in range(0, 30):
        solutions.append(generate_solution())
    return solutions


def fitness(solution):
    count = 0
    for i in solution:
        if i == '1':
            count += 1
    return count


def evolutionary_algorithm():
    population = generate_initial_population()
    for solution in population:
        # select fitter individuals (selection)
        # mutation
        # crossover
        # reevaluate fitness
        # generate new population
        print(solution)


if __name__ == '__main__':
    evolutionary_algorithm()
