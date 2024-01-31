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

def post_process_solution(solution, items, capacity):
    bins = {}
    for idx, bin_num in enumerate(solution):
        item_weight = items[idx % len(items)]
        if bin_num in bins:
            bins[bin_num].append(item_weight)
        else:
            bins[bin_num] = [item_weight]
    fitness = calculate_fitness(solution, items, capacity)
    print(f"Solution Fitness: {fitness}")
    for bin_num, contents in sorted(bins.items()):
        print(f"Bin {bin_num}: Items {contents}, Total Weight: {sum(contents)}")

# Replace this string with your actual problem instances
problem_text = """
'BPP      1'
      46
    1000
     200         3
     199         1
     198         2
     197         2
     194         2
     193         1
     192         1
     191         3
     190         2
     189         1
     188         2
     187         2
     186         1
     185         4
     184         3
     183         3
     182         3
     181         2
     180         1
     179         4
     178         1
     177         4
     175         1
     174         1
     173         2
     172         1
     171         3
     170         2
     169         3
     167         2
     165         2
     164         1
     163         4
     162         1
     161         1
     160         2
     159         1
     158         3
     157         1
     156         6
     155         3
     154         2
     153         1
     152         3
     151         2
     150         4
'BPP      2'
      47
    1000
     200         2
     199         4
     198         1
     197         1
     196         2
     195         2
     194         2
     193         1
     191         2
     190         1
     189         2
     188         1
     187         2
     186         1
     185         2
     184         5
     183         1
     182         1
     181         3
     180         2
     179         2
     178         1
     176         1
     175         2
     174         5
     173         1
     172         3
     171         1
     170         4
     169         2
     168         1
     167         5
     165         2
     164         2
     163         3
     162         2
     160         2
     159         2
     158         2
     157         4
     156         3
     155         2
     154         1
     153         3
     152         2
     151         2
     150         2
'BPP      3'
      44
    1000
     200         1
     199         2
     197         2
     196         2
     193         3
     192         2
     191         2
     190         2
     189         3
     188         1
     187         1
     185         3
     183         2
     182         1
     181         3
     180         3
     179         3
     178         1
     177         5
     176         2
     175         5
     174         4
     173         1
     171         3
     170         1
     169         2
     168         5
     167         1
     166         4
     165         2
     163         1
     162         2
     161         2
     160         3
     159         2
     158         2
     157         1
     156         3
     155         3
     154         1
     153         2
     152         3
     151         2
     150         1
'BPP      4'
      42
    1000
     200         3
     199         5
     198         4
     197         1
     195         1
     193         4
     192         1
     188         1
     187         1
     186         3
     185         3
     184         2
     183         2
     182         1
     181         1
     180         3
     179         2
     178         6
     177         2
     176         4
     175         1
     173         4
     172         4
     170         1
     169         3
     168         4
     167         1
     165         3
     164         1
     163         2
     162         4
     161         1
     160         3
     159         3
     158         1
     157         3
     155         2
     154         3
     153         1
     152         3
     151         1
     150         1
'BPP      5'
      44
    1000
     200         5
     199         2
     198         2
     197         2
     196         1
     195         3
     194         2
     193         2
     192         4
     191         2
     190         4
     188         3
     187         2
     186         2
     185         1
     184         1
     183         1
     182         1
     181         3
     180         1
     178         3
     177         2
     176         2
     174         1
     173         1
     172         1
     171         3
     168         2
     167         1
     165         1
     164         1
     163         1
     162         3
     161         3
     160         3
     159         2
     158         3
     157         3
     156         2
     155         5
     154         3
     153         3
     151         5
     150         2
"""

if __name__ == "__main__":
    problem_instances = parse_problem_instances(problem_text)
    plt.figure(figsize=(10, 6))

    for problem_instance in problem_instances:
        best_solution, avg_fitness_over_generations = run_ga(problem_instance)
        plt.plot(avg_fitness_over_generations, label=f"{problem_instance['name']}")

        # Post-process and print the best solution details
        print(f"\nBest Solution for {problem_instance['name']}:")
        items = sum(([weight] * count for weight, count in problem_instance['items']), [])
        post_process_solution(best_solution, items, problem_instance['capacity'])

    plt.title("Average Best Fitness over Generations for All Problem Instances")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.legend()
    plt.show()
