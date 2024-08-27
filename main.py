import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.hidden = np.tanh(np.dot(X, self.weights1))
        self.output = np.tanh(np.dot(self.hidden, self.weights2))
        return self.output
    
    def mutate(self, mutation_rate):
        self.weights1 += np.random.randn(*self.weights1.shape) * mutation_rate
        self.weights2 += np.random.randn(*self.weights2.shape) * mutation_rate

class LemonadeStandEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.day = 0
        self.money = 2000  # cents
        self.inventory = {'cups': 0, 'lemons': 0, 'sugar': 0, 'ice': 0}
        self.weather = random.choice(['sunny', 'cloudy', 'rainy'])
        self.temperature = random.randint(60, 100)  # Fahrenheit
        self.reputation = 1.0
        self.rep_level = 1

    def step(self, action):
        cups_to_buy, lemons_to_buy, sugar_to_buy, ice_to_buy, price, recipe_lemons, recipe_sugar, recipe_ice = action
        
        # Ensure recipe amounts are integers between 1 and 10
        recipe_lemons = max(1, min(10, int(recipe_lemons)))
        recipe_sugar = max(1, min(10, int(recipe_sugar)))
        recipe_ice = max(1, min(10, int(recipe_ice)))
        recipe = [recipe_lemons, recipe_sugar, recipe_ice]
        
        # Perform purchases
        total_cost = (cups_to_buy * 5 + lemons_to_buy * 10 + sugar_to_buy * 8 + ice_to_buy * 5)
        if total_cost <= self.money:
            self.money -= total_cost
            self.inventory['cups'] += cups_to_buy
            self.inventory['lemons'] += lemons_to_buy
            self.inventory['sugar'] += sugar_to_buy
            self.inventory['ice'] += ice_to_buy
        
        # Simulate sales
        max_lemonade = min(self.inventory['cups'], 
                           self.inventory['lemons'] // recipe_lemons,
                           self.inventory['sugar'] // recipe_sugar,
                           self.inventory['ice'] // recipe_ice)
        
        demand = self.calculate_demand(price, self.temperature, self.weather, recipe)
        sales = min(max_lemonade, demand)
        
        # Update inventory and money
        self.inventory['cups'] -= sales
        self.inventory['lemons'] -= sales * recipe_lemons
        self.inventory['sugar'] -= sales * recipe_sugar
        self.inventory['ice'] -= sales * recipe_ice
        self.money += sales * price
        
        # Update reputation
        self.update_reputation(price, recipe)
        
        # Move to next day
        self.day += 1
        self.weather = random.choice(['sunny', 'cloudy', 'rainy'])
        self.temperature = random.randint(60, 100)
        
        return self.money

    def calculate_demand(self, price, temperature, weather, recipe):
        weather_index = {'sunny': 0, 'cloudy': 2, 'rainy': 4}[weather]
        
        base_demand = ((temperature - 50) / 200 + (5 - weather_index) / 20) * \
                      (((temperature / 4) - price) / (temperature / 4) + 1)
        
        if self.rep_level < random.random() * (self.rep_level - 500):
            base_demand *= self.reputation
        
        base_demand *= (recipe[0] + 1) / 5  # Lemons factor
        base_demand *= (recipe[1] + 4) / 8  # Sugar factor
        
        # Add some randomness
        demand = (base_demand + random.uniform(-0.1, 0.1)) * 1.3
        
        return max(0, int(demand * 100))  # Multiply by 100 to align with the original scale

    def update_reputation(self, price, recipe):
        opinion = 0.8 + random.random() * 0.4
        opinion *= recipe[0] / 4  # Lemons factor
        opinion *= recipe[1] / 4  # Sugar factor
        opinion *= recipe[2] / ((self.temperature - 50) / 5) + 1  # Ice factor
        opinion *= ((self.temperature - 50) / 5 + 1) / (recipe[2] + 4)  # Temperature-Ice balance
        opinion *= (self.temperature / 4 - price) / (self.temperature / 4) + 1  # Price factor
        
        opinion = max(0, min(opinion, 2))
        self.reputation += opinion
        self.rep_level += 1

    def get_state(self):
        return np.array([
            self.temperature / 100,  # Normalize temperature
            int(self.weather == 'sunny'),
            int(self.weather == 'cloudy'),
            int(self.weather == 'rainy'),
            self.inventory['cups'] / 100,  # Normalize inventory
            self.inventory['lemons'] / 100,
            self.inventory['sugar'] / 100,
            self.inventory['ice'] / 100,
            self.money / 10000,  # Normalize money
            self.reputation / 10  # Normalize reputation
        ])

def evolve_population(population, fitness_scores, mutation_rate):
    new_population = []
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    
    # Elitism: keep the best performer
    new_population.append(sorted_population[0])
    
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(sorted_population[:len(population)//2], 2)
        child = NeuralNetwork(parent1.weights1.shape[0], parent1.weights1.shape[1], parent1.weights2.shape[1])
        
        # Crossover
        child.weights1 = (parent1.weights1 + parent2.weights1) / 2
        child.weights2 = (parent1.weights2 + parent2.weights2) / 2
        
        # Mutation
        child.mutate(mutation_rate)
        
        new_population.append(child)
    
    return new_population

# Hyperparameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
SIMULATION_DAYS = 30

# Initialize population
input_size = 9  # temperature, weather(3), cups, lemons, sugar, ice, money
hidden_size = 10
output_size = 8  # cups_to_buy, lemons_to_buy, sugar_to_buy, ice_to_buy, price, recipe_lemons, recipe_sugar, recipe_ice
population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(POPULATION_SIZE)]

# Training loop
for generation in range(GENERATIONS):
    fitness_scores = []
    
    for nn in population:
        env = LemonadeStandEnvironment()
        for _ in range(SIMULATION_DAYS):
            state = env.get_state()
            action = nn.forward(state)
            action = (action + 1) / 2  # Scale from [-1, 1] to [0, 1]
            action[:4] *= 100  # Scale purchase amounts
            action[4] *= 100  # Scale price
            action[5:] = np.round(action[5:] * 10)  # Scale recipe amounts to integers 0-10
            reward = env.step(action)
        
        fitness_scores.append(reward)
    
    print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}, Avg Fitness: {sum(fitness_scores) / len(fitness_scores)}")
    
    population = evolve_population(population, fitness_scores, MUTATION_RATE)

# Test the best strategy
best_nn = population[fitness_scores.index(max(fitness_scores))]
env = LemonadeStandEnvironment()
total_profit = 0

for day in range(SIMULATION_DAYS):
    state = env.get_state()
    action = best_nn.forward(state)
    action = (action + 1) / 2
    action[:4] *= 100
    action[4] *= 100
    action[5:] = np.round(action[5:] * 10)  # Ensure recipe amounts are integers 0-10
    reward = env.step(action)
    total_profit += reward - 2000  # Subtract initial money

    print(f"Day {day + 1}: Weather: {env.weather}, Temp: {env.temperature}°F")
    print(f"Action: Buy - Cups: {action[0]:.0f}, Lemons: {action[1]:.0f}, Sugar: {action[2]:.0f}, Ice: {action[3]:.0f}")
    print(f"Price: ${action[4]/100:.2f}, Recipe - Lemons: {action[5]:.0f}, Sugar: {action[6]:.0f}, Ice: {action[7]:.0f}")
    print(f"Money: ${reward/100:.2f}")
    print("---")

print(f"Total Profit over {SIMULATION_DAYS} days: ${total_profit/100:.2f}")