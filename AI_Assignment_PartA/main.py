import random
import matplotlib.pyplot as plt

# Parameters
solution_size = 30
population_size = 100
generations = 100
mutation_rate = 0.01
elite_size = 2

# Common Genetic Algorithm Functions
def generate_solution(solution_size):
    return ''.join(random.choice('01') for _ in range(solution_size))


def generate_population(population_size, solution_size):
    return [generate_solution(solution_size) for _ in range(population_size)]


def mutate(solution):
    return ''.join('1' if bit == '0' and random.random() < mutation_rate else
                   '0' if bit == '1' and random.random() < mutation_rate else
                   bit for bit in solution)


def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    return random.choices(population, weights=selection_probs, k=len(population))


def elitism(population, fitness_scores):
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elite = [individual for individual, _ in sorted_population[:elite_size]]
    return elite


# Problem-Specific Fitness Functions
def fitness_one_max(solution):
    return sum(bit == '1' for bit in solution)


def fitness_target_string(solution, target):
    return sum(s == t for s, t in zip(solution, target))


def fitness_deceptive(solution):
    count_ones = sum(bit == '1' for bit in solution)
    return 2 * len(solution) if count_ones == 0 else count_ones


# Main Genetic Algorithm Execution
def run_genetic_algorithm(fitness_func, additional_args=None):
    population = generate_population(population_size, solution_size)
    average_fitness_history = []

    for _ in range(generations):
        fitness_scores = [fitness_func(individual, *additional_args) if additional_args else fitness_func(individual) for individual in population]
        elite = elitism(population, fitness_scores)
        selected_population = roulette_wheel_selection(population, fitness_scores)
        selected_population = selected_population[:-elite_size]

        new_population = elite
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = one_point_crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:population_size]
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        average_fitness_history.append(average_fitness)

    return average_fitness_history


# Running for each problem
# 1.1 One-max Problem
one_max_history = run_genetic_algorithm(fitness_one_max)

# 1.2 Target String Problem
target_string = "11001010101100101010"
target_string_history = run_genetic_algorithm(fitness_target_string, [target_string])

# 1.3 Deceptive Landscape
deceptive_history = run_genetic_algorithm(fitness_deceptive)

# Plotting the results for each problem
plt.figure(figsize=(12, 8))
plt.plot(one_max_history, label='One-max Problem')
plt.plot(target_string_history, label='Target String Problem')
plt.plot(deceptive_history, label='Deceptive Landscape')
plt.title('Average Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.legend()
plt.grid(True)
plt.savefig('genetic_algorithm_results.png', bbox_inches='tight')
plt.show()
