import random

# Parameters
solution_size = 30
population_size = 100
generations = 10
mutation_rate = 0.01
elite_size = 2


def generate_solution():
    return ''.join(random.choice('01') for _ in range(solution_size))


def generate_initial_population():
    return [generate_solution() for _ in range(population_size)]


def fitness(solution):
    return sum(bit == '1' for bit in solution)


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    return random.choices(population, weights=selection_probs, k=len(population))


def elitism(population, fitness_scores):
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elite = [individual for individual, _ in sorted_population[:elite_size]]
    return elite


def evolutionary_algorithm():
    population = generate_initial_population()

    for _ in range(generations):
        # calculate fitness for generation
        # select new population sample
        # elitism & roulette

        # create new population with crossover and mutation
        for solution in population:
            # mutation & crossover
            # reevaluate population fitness
            # average fitness of the population
            print(solution)
            print(fitness(solution))


if __name__ == '__main__':
    evolutionary_algorithm()
