import random
import matplotlib.pyplot as plt

population_size = 50
generations = 100
mutation_rate = 0.1

class Bin:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add_item(self, item):
        if self.can_add(item):
            self.items.append(item)
            return True
        return False

    def can_add(self, item):
        return sum(self.items) + item <= self.capacity

    def remaining_capacity(self):
        return self.capacity - sum(self.items)

    def __str__(self):
        return f"Bin(capacity={self.capacity}, items={self.items})"


def initialize_population(population_size, items, bin_capacity):
    population = []
    for _ in range(population_size):
        bins = []
        for item in items:
            placed = False
            for bin in bins:
                if bin.add_item(item):
                    placed = True
                    break
            if not placed:
                new_bin = Bin(bin_capacity)
                new_bin.add_item(item)
                bins.append(new_bin)
        population.append(bins)
    return population

def read_binpacking_problems(file_path):
    problems = []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("'BPP"):  # check for problem name
            name = lines[i]
            m = int(lines[i + 1])
            capacity = int(lines[i + 2])
            items = []
            for j in range(m):
                line = i + 3 + j
                weight, count = map(int, lines[line].split())
                for _ in range(count):
                    items.append(weight)
            problems.append((name, capacity, items))
            i += 3 + m  # move to the next problem
        else:
            i += 1  # increment if line is not new problem

    return problems


def calculate_fitness(bins):
    return len(bins)

def tournament_selection(population, k=3):
    selected = []
    for _ in range(len(population)):
        aspirants = random.sample(population, k)
        winner = min(aspirants, key=calculate_fitness)
        selected.append(winner)
    return selected

def mutate(bins):
    if len(bins) > 1:
        bin1, bin2 = random.sample(bins, 2)
        if bin1.items and bin2.items:
            item1, item2 = random.choice(bin1.items), random.choice(bin2.items)
            if bin1.can_add(item2) and bin2.can_add(item1):
                bin1.items.remove(item1)
                bin2.items.remove(item2)
                bin1.add_item(item2)
                bin2.add_item(item1)
    return bins

def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def genetic_algorithm(items, bin_capacity, population_size, generations):
    population = initialize_population(population_size, items, bin_capacity)
    avg_fitness_evolution = []  # List to store average fitness per generation

    for generation in range(generations):
        fitness_scores = [calculate_fitness(solution) for solution in population]
        parents = tournament_selection(population)

        offspring = []
        for i in range(0, len(parents), 2):
            for child in one_point_crossover(parents[i], parents[(i + 1) % len(parents)]):
                offspring.append(mutate(child))

        population = offspring[:population_size]

        # Calculate and append average fitness for this generation
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_fitness_evolution.append(avg_fitness)

    best_solution = min(population, key=calculate_fitness)
    return best_solution, calculate_fitness(best_solution), avg_fitness_evolution

avg_fitness_per_problem = []  # List to store average fitness per problem

bins_used = []

problems = read_binpacking_problems('Binpacking.txt')
for name, capacity, items in problems:
    best_solution, best_fitness, avg_fitness_evolution = genetic_algorithm(items, capacity, population_size, generations)
    print(f"Best solution for {name}: {best_fitness} bins used")
    bins_used.append(best_fitness)
    avg_fitness_per_problem.append(avg_fitness_evolution)

# Plotting average fitness evolution per problem
plt.figure(figsize=(12, 6))
for i, avg_fitness_evolution in enumerate(avg_fitness_per_problem):
    plt.plot(range(generations), avg_fitness_evolution, label=f'Problem {i+1}')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Average Fitness Evolution per Problem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
