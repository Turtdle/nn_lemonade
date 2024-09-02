import random
import json

# Assuming the lemonade.py module is in the same directory
from lemonade import simulate_day, WEATHER

class Chromosome:
    def __init__(self):
        self.genes = {
            'price': random.randint(5, 99),
            'recipe_lemons': random.randint(4, 12),
            'recipe_sugar': random.randint(4, 12),
            'recipe_ice': random.randint(1, 15),
            'buy_cups': random.randint(0, 500),
            'buy_lemons': random.randint(0, 750),
            'buy_sugar': random.randint(0, 750),
            'buy_ice': random.randint(0, 1500)
        }
        self.fitness = 0

    def mutate(self):
        gene = random.choice(list(self.genes.keys()))
        if gene == 'price':
            self.genes[gene] += random.randint(-5, 5)
        elif gene in ['recipe_lemons', 'recipe_sugar']:
            self.genes[gene] += random.randint(-1, 1)
            self.genes[gene] = max(4, self.genes[gene])  # Ensure minimum of 4
        elif gene == 'recipe_ice':
            self.genes[gene] += random.randint(-2, 2)
        else:  # buy amounts
            self.genes[gene] += random.randint(-10, 10)
        
        self.genes[gene] = max(1, self.genes[gene])  # Ensure all values are at least 1

def crossover(parent1, parent2):
    child = Chromosome()
    for gene in child.genes:
        if random.random() < 0.5:
            child.genes[gene] = parent1.genes[gene]
        else:
            child.genes[gene] = parent2.genes[gene]
    return child

def evaluate_fitness(chromosome, fixed_temperature, fixed_weather):
    state = {
        'cups': 0,
        'lemons': 0,
        'sugar': 0,
        'ice': 0,
        'money': 2000,  # Starting with $20
        'temperature': fixed_temperature,
        'weather': fixed_weather,
        'price': chromosome.genes['price'],
        'recipe_lemons': chromosome.genes['recipe_lemons'],
        'recipe_sugar': chromosome.genes['recipe_sugar'],
        'recipe_ice': chromosome.genes['recipe_ice'],
        'total_income': 0,
        'rep_level': 0,
        'reputation': 0,
        'buy_order': [
            chromosome.genes['buy_cups'],
            chromosome.genes['buy_lemons'],
            chromosome.genes['buy_sugar'],
            chromosome.genes['buy_ice']
        ],
        'failed_to_buy': False
    }

    for _ in range(7):  # Simulate 7 days
        state = simulate_day(state)

    return state['money'] - 2000  # Return profit

def genetic_algorithm(population_size=750, generations=100, fixed_temperature=75, fixed_weather='sunny'):
    population = [Chromosome() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        for chromosome in population:
            chromosome.fitness = evaluate_fitness(chromosome, fixed_temperature, fixed_weather)

        # Check if all chromosomes have fitness 0
        if all(chromosome.fitness == 0 for chromosome in population):
            # Reroll the entire generation
            population = [Chromosome() for _ in range(population_size)]
            print(f"Generation {generation + 1}: All chromosomes had fitness 0. Rerolling entire generation.")
            continue  # Skip to the next generation

        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Print best fitness of this generation
        print(f"Generation {generation + 1}, Best Fitness: {population[0].fitness}")

        # Select top half as parents for next generation
        parents = population[:population_size // 2]

        # Create next generation
        next_generation = parents.copy()
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # 10% chance of mutation
                child.mutate()
            next_generation.append(child)

        population = next_generation

    return population[0]  # Return the best chromosome

def run_optimization_for_conditions():
    temperatures = [60, 70, 80, 90, 100]
    results = {}

    for temp in temperatures:
        for weather in WEATHER:
            print(f"\nRunning optimization for Temperature: {temp}°F, Weather: {weather}")
            best_solution = genetic_algorithm(population_size=750, generations=150, fixed_temperature=temp, fixed_weather=weather)
            
            results[(temp, weather)] = {
                'genes': best_solution.genes,
                'fitness': best_solution.fitness
            }
            print(f"Best solution for Temperature: {temp}°F, Weather: {weather}:")
            print(best_solution.genes)

    return results

# Run the optimization for various conditions
print("Running genetic algorithm for various weather conditions and temperatures...")
optimization_results = run_optimization_for_conditions()

# Save results to a file
with open('lemonade_optimization_results.json', 'w') as f:
    json_results = {}
    for (temp, weather), result in optimization_results.items():
        key = f"{temp}_{weather}"
        json_results[key] = {
            'temperature': temp,
            'weather': weather,
            'genes': result['genes'],
            'profit': result['fitness'] / 100  # Convert to dollars
        }
    json.dump(json_results, f, indent=4)

print("Results have been saved to 'lemonade_optimization_results.json'")