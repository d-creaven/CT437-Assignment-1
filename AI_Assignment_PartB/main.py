import random
import matplotlib.pyplot as plt

# GA Parameters
POP_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.02
GENERATIONS = 200


def parse_problem_instances(text):
    instances = []
    lines = text.strip().split('\n')
    i = 0
    while i < len(lines):
        name = lines[i].strip().strip("'")
        i += 1
        m = int(lines[i].strip())
        i += 1
        capacity = int(lines[i].strip())
        i += 1
        items = []
        for _ in range(m):
            weight, count = map(int, lines[i].strip().split())
            items.append((weight, count))
            i += 1
        instances.append({'name': name, 'capacity': capacity, 'items': items})
    return instances


def initialize_population(problem_instance):
    population = []
    for _ in range(POP_SIZE):
        individual = []
        for weight, count in problem_instance['items']:
            for _ in range(count):
                individual.append(random.randint(1, POP_SIZE))  # Assign to a random bin
        random.shuffle(individual)  # Shuffle to ensure randomness
        population.append(individual)
    return population


def calculate_fitness(solution, items, capacity):
    bins = {}
    for idx, bin_num in enumerate(solution):
        item_weight = items[idx]  # Directly use the item weight
        bins.setdefault(bin_num, 0)
        bins[bin_num] += item_weight
    penalty = sum(1 for total_weight in bins.values() if total_weight > capacity)
    fitness = len(bins) + penalty
    return fitness


def select_parents(population, capacity, items):
    parents = []
    for _ in range(2):
        contenders = random.sample(population, 3)
        contenders.sort(key=lambda ind: calculate_fitness(ind, items, capacity))
        parents.append(contenders[0])
    return parents


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 2)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1[:], parent2[:]


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.randint(1, POP_SIZE)


def run_ga(problem_instance):
    capacity = problem_instance['capacity']
    items = sum(([weight] * count for weight, count in problem_instance['items']), [])
    population = initialize_population(problem_instance)
    best_fitness = float('inf')
    avg_fitness_over_generations = []

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = select_parents(population, capacity, items)
            offspring1, offspring2 = crossover(parent1, parent2)
            mutate(offspring1)
            mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        population = sorted(new_population, key=lambda ind: calculate_fitness(ind, items, capacity))[:POP_SIZE]
        current_best_fitness = calculate_fitness(population[0], items, capacity)
        best_fitness = min(best_fitness, current_best_fitness)
        avg_fitness = sum(calculate_fitness(ind, items, capacity) for ind in population) / POP_SIZE
        avg_fitness_over_generations.append(avg_fitness)

    return population[0], avg_fitness_over_generations


def parse_problem_instances_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return parse_problem_instances(text)


if __name__ == "__main__":
    file_path = 'Binpacking.txt'
    problem_instances = parse_problem_instances_from_file(file_path)
    plt.figure(figsize=(10, 6))

    for problem_instance in problem_instances:
        best_solution, avg_fitness_over_generations = run_ga(problem_instance)
        plt.plot(avg_fitness_over_generations, label=f"{problem_instance['name']}")

        print(f"\nBest Solution for {problem_instance['name']}:")
        bins = {}
        for item_idx, bin_num in enumerate(best_solution):
            if bin_num not in bins:
                bins[bin_num] = []
            # Assuming each item's weight is repeated according to its count in the problem instance
            item_weight = problem_instance['items'][item_idx % len(problem_instance['items'])][0]
            bins[bin_num].append(item_weight)

        # Print the total number of bins used
        print(f"Total number of bins used: {len(bins)}")

        for bin_num, items in bins.items():
            print(f"  Bin {bin_num}: Items {items}, Total Weight: {sum(items)}")

    plt.title("Average Best Fitness over Generations for All Problem Instances")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig('bin_packing_results.png', bbox_inches='tight')
    plt.show()
