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
            'buy_lemons': random.randint(0, 500),
            'buy_sugar': random.randint(0, 500),
            'buy_ice': random.randint(0, 1500)
        }

class Individual:
    def __init__(self):
        self.chromosomes = [Chromosome() for _ in range(7)]  # 7 chromosomes, one for each day
        self.fitness = 0

    def mutate(self):
        chromosome = random.choice(self.chromosomes)
        gene = random.choice(list(chromosome.genes.keys()))
        if gene == 'price':
            chromosome.genes[gene] += random.randint(-3, 3)
        elif gene in ['recipe_lemons', 'recipe_sugar']:
            chromosome.genes[gene] += random.randint(-1, 1)
            chromosome.genes[gene] = max(4, chromosome.genes[gene])  # Ensure minimum of 4
        elif gene == 'recipe_ice':
            chromosome.genes[gene] += random.randint(-2, 2)
        else:  # buy amounts
            chromosome.genes[gene] += random.randint(-5, 5)
        
        chromosome.genes[gene] = max(1, chromosome.genes[gene])  # Ensure all values are at least 1

def crossover(parent1, parent2):
    child = Individual()
    for i in range(7):
        if random.random() < 0.5:
            child.chromosomes[i] = parent1.chromosomes[i]
        else:
            child.chromosomes[i] = parent2.chromosomes[i]
    return child

def evaluate_fitness(individual, fixed_temperature, fixed_weather):
    state = {
        'cups': 0,
        'lemons': 0,
        'sugar': 0,
        'ice': 0,
        'money': 2000,  # Starting with $20
        'temperature': fixed_temperature,
        'weather': fixed_weather,
        'total_income': 0,
        'rep_level': 0,
        'reputation': 0,
        'failed_to_buy': False
    }

    for chromosome in individual.chromosomes:
        state['price'] = chromosome.genes['price']
        state['recipe_lemons'] = chromosome.genes['recipe_lemons']
        state['recipe_sugar'] = chromosome.genes['recipe_sugar'] 
        state['recipe_ice'] = chromosome.genes['recipe_ice']
        state['buy_order'] = [
            chromosome.genes['buy_cups'],
            chromosome.genes['buy_lemons'],
            chromosome.genes['buy_sugar'],
            chromosome.genes['buy_ice']
        ]

        state = simulate_day(state)
        state['ice'] = 0  # Reset ice at the end of each day
        state['customers'] = []
        state['failed_to_buy'] = False

    return state['money'] - 2000 + state['total_sold'] * 10, state  # Return profit and final state

def genetic_algorithm(population_size=750, generations=100, fixed_temperature=75, fixed_weather='sunny'):
    population = [Individual() for _ in range(population_size)]
    past_best_fitness = -1
    for generation in range(generations):
        # Evaluate fitness
        for individual in population:
            individual.fitness, _ = evaluate_fitness(individual, fixed_temperature, fixed_weather)

        # Check if all individuals have fitness 0
        if all(individual.fitness <= 0 for individual in population):
            # Reroll the entire generation
            population = [Individual() for _ in range(population_size)]
            print(f"Generation {generation + 1}: All individuals had fitness 0. Rerolling entire generation.")
            continue  # Skip to the next generation

            
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Print best fitness of this generation
        print(f"Generation {generation + 1}, Best Fitness: {population[0].fitness}")
        print(f"Average Fitness: {sum(individual.fitness for individual in population) / population_size}")

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

    return population[0]  # Return the best individual

def run_optimization_for_conditions():
    temperatures = [80]
    results = {}

    for temp in temperatures:
        for weather in ['sunny']:
            print(f"\nRunning optimization for Temperature: {temp}°F, Weather: {weather}")
            best_solution = genetic_algorithm(population_size=100, generations=500, fixed_temperature=temp, fixed_weather=weather)
            
            _, final_state = evaluate_fitness(best_solution, temp, weather)  # Get the final state
            
            results[(temp, weather)] = {
                'chromosomes': [chromosome.genes for chromosome in best_solution.chromosomes],
                'fitness': best_solution.fitness,
                'final_state': final_state
            }
            print(f"Best solution for Temperature: {temp}°F, Weather: {weather}:")
            print([chromosome.genes for chromosome in best_solution.chromosomes])

    return results

# Run the optimization for various conditions
print("Running genetic algorithm for various weather conditions and temperatures...")
optimization_results = run_optimization_for_conditions()
print("Optimization complete! Best solutions found:")
print(optimization_results)

# Perform a final run with the best solution and include all state results in the JSON file
best_solution_key = list(optimization_results.keys())[0]  # Get the key of the first (and only) best solution
best_solution = Individual()
best_solution.chromosomes = [Chromosome() for _ in range(7)]
for i, genes in enumerate(optimization_results[best_solution_key]['chromosomes']):
    best_solution.chromosomes[i].genes = genes

print("\nPerforming a final run with the best solution...")
_, final_state = evaluate_fitness(best_solution, best_solution_key[0], best_solution_key[1])
final_results = {
    'chromosomes': [chromosome.genes for chromosome in best_solution.chromosomes],
    'final_state': final_state
}

# Save results to a file
with open('lemonade_optimization_results.json', 'w') as f:
    json_results = {}
    for (temp, weather), result in optimization_results.items():
        key = f"{temp}_{weather}"
        json_results[key] = {
            'temperature': temp,
            'weather': weather,
            'chromosomes': result['chromosomes'],
            'profit': result['fitness'] / 100,  # Convert to dollars
            'final_state': result['final_state']
        }
    json_results['final_results'] = {
        'chromosomes': final_results['chromosomes'],
        'final_state': final_results['final_state']
    }
    json.dump(json_results, f, indent=4)

print("Results have been saved to 'lemonade_optimization_results.json'")