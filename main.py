import numpy as np
from keras.models import clone_model
from lemonade import LemonadeStand, format_money
from neural_net import build_model
import time
import logging
from datetime import datetime
class EvolutionaryTrainer:
    def __init__(self, population_size=50, generations=5, mutation_rate=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        print(f"Initializing population with {population_size} models...")
        self.population = [build_model() for _ in range(population_size)]
        print("Population initialized.")

    def mutate(self, model):
        new_model = clone_model(model)
        new_model.set_weights(model.get_weights())
        weights = new_model.get_weights()
        for i in range(len(weights)):
            if np.random.random() < self.mutation_rate:
                noise = np.random.normal(0, 0.1, size=weights[i].shape)
                weights[i] += noise
        new_model.set_weights(weights)
        return new_model

    def crossover(self, model1, model2):
        new_model = clone_model(model1)
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        new_weights = []
        for w1, w2 in zip(weights1, weights2):
            mask = np.random.random(size=w1.shape) >= 0.5
            new_w = np.where(mask, w1, w2)
            new_weights.append(new_w)
        new_model.set_weights(new_weights)
        return new_model

    def evaluate_model(self, model):
        game = LemonadeStand(duration=7)
        total_profit = 0
        total_revenue = 0
        total_expenses = 0
        for _ in range(game.state['duration']):
            game.new_day()
            state = np.array([
                float(game.state['temperature']) / 40,  # Normalized
                float(game.state['price_cups'][0]) / 100,
                float(game.state['price_lemons'][0]) / 100,
                float(game.state['price_sugar'][0]) / 100,
                float(game.state['price_ice'][0]) / 100,
                float(game.state['cups']) / 1000,
                float(game.state['lemons']) / 1000,
                float(game.state['sugar']) / 1000,
                float(game.state['ice']) / 1000,
                float(game.state['money']) / 10000,
                float(game.state['popularity']) / 100
            ])
            prediction = model.predict(np.array([state]), verbose=0)[0]
            
            # Convert predictions to purchase amounts
            cups_to_buy = max(0, int(prediction[0] * 50))  # Scale to 0-50 range
            lemons_to_buy = max(0, int(prediction[1] * 20))  # Scale to 0-20 range
            sugar_to_buy = max(0, int(prediction[2] * 20))  # Scale to 0-20 range
            ice_to_buy = max(0, int(prediction[3] * 50))  # Scale to 0-50 range
            
            # Calculate costs for each item
            cups_cost = cups_to_buy * game.state['price_cups'][0]
            lemons_cost = lemons_to_buy * game.state['price_lemons'][0]
            sugar_cost = sugar_to_buy * game.state['price_sugar'][0]
            ice_cost = ice_to_buy * game.state['price_ice'][0]
            
            total_cost = cups_cost + lemons_cost + sugar_cost + ice_cost
            
            # Apply proportional scaling if total cost exceeds available money
            if total_cost > game.state['money']:
                scale_factor = game.state['money'] / total_cost
                cups_to_buy = max(0, int(cups_to_buy * scale_factor))
                lemons_to_buy = max(0, int(lemons_to_buy * scale_factor))
                sugar_to_buy = max(0, int(sugar_to_buy * scale_factor))
                ice_to_buy = max(0, int(ice_to_buy * scale_factor))
            
            # Buy supplies
            game.buy_supplies('cups', cups_to_buy, 0)
            game.buy_supplies('lemons', lemons_to_buy, 0)
            game.buy_supplies('sugar', sugar_to_buy, 0)
            game.buy_supplies('ice', ice_to_buy, 0)
            
            price = max(1, min(prediction[4] * 2, 200))  # Price between $0.01 and $2.00
            game.set_price(price)
            
            recipe_amount1 = max(1, int(prediction[5] * 10))  # Scale to 1-10 range
            recipe_amount2 = max(1, int(prediction[6] * 10))  # Scale to 1-10 range
            recipe_amount3 = max(1, int(prediction[7] * 10))  # Scale to 1-10 range
            game.set_recipe(recipe_amount1, recipe_amount2, recipe_amount3)
            
            results = game.simulate_day()
            daily_revenue = results['total_sold'] * game.state['price']
            daily_expenses = game.state['total_expenses']
            daily_profit = daily_revenue - daily_expenses
            
            total_revenue += daily_revenue
            total_expenses += daily_expenses
            total_profit += daily_profit
        
        # Calculate inventory diversity score
        inventory = np.array([game.state['cups'], game.state['lemons'], game.state['sugar'], game.state['ice']])
        total_inventory = np.sum(inventory)
        if total_inventory > 0:
            inventory_ratios = inventory / total_inventory
            diversity_score = -np.sum(inventory_ratios * np.log(inventory_ratios + 1e-10))  # Shannon entropy
        else:
            diversity_score = 0
        
        # Calculate the final fitness score
        profit_score = max(0, total_profit) * 100000
        revenue_score = total_revenue * 100  # Encourage higher revenue
        inventory_score = total_inventory * 0.05  # Slight bonus for total inventory
        diversity_reward = diversity_score * 1000  # Scale the diversity score
        expense_penalty = max(0, total_expenses - total_revenue) * 0.05  # Penalty for overspending, but not as harsh
        
        fitness_score = profit_score + revenue_score + inventory_score + diversity_reward - expense_penalty
        
        return fitness_score

    def train(self):
        print(f"Starting training for {self.generations} generations...")
        start_time = time.time()
        for generation in range(self.generations):
            gen_start_time = time.time()
            print(f"\nEvaluating Generation {generation + 1}/{self.generations}")
            fitness_scores = []
            for i, model in enumerate(self.population):
                fitness = self.evaluate_model(model)
                fitness_scores.append(fitness)
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i + 1}/{self.population_size} models...")
            print(fitness_scores)
            print(f"Generation {generation + 1} evaluation complete.")
            print(f"Best fitness: {format_money(max(fitness_scores))}")
            print(f"Average fitness: {format_money(sum(fitness_scores) / len(fitness_scores))}")
            
            elite_size = self.population_size // 5
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite_population = [self.population[i] for i in elite_indices]
            
            print("Creating new population...")
            new_population = elite_population.copy()
            while len(new_population) < self.population_size:
                if np.random.random() < 0.3:
                    parent = np.random.choice(elite_population)
                    child = self.mutate(parent)
                else:
                    parents = np.random.choice(elite_population, 2, replace=False)
                    child = self.crossover(parents[0], parents[1])
                new_population.append(child)
            
            self.population = new_population
            gen_end_time = time.time()
            print(f"Generation {generation + 1} completed in {gen_end_time - gen_start_time:.2f} seconds.")

        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

        print("Evaluating final population...")
        final_fitness_scores = [self.evaluate_model(model) for model in self.population]
        best_model = self.population[np.argmax(final_fitness_scores)]
        print(f"Best model final fitness: {format_money(max(final_fitness_scores))}")
        return best_model

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a unique filename based on the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"lemonade_stand_log_{current_time}.txt"

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print("Initializing Evolutionary Trainer...")
    trainer = EvolutionaryTrainer()
    print("Beginning training process...")
    best_model = trainer.train()
    
    print("\nTesting best model on a 30-day game simulation...")
    game = LemonadeStand(duration=30)
    total_profit = 0
    for day in range(game.state['duration']):
        game.new_day()
        state = np.array([
            float(game.state['temperature']),
            float(game.state['price_cups'][0]),
            float(game.state['price_lemons'][0]),
            float(game.state['price_sugar'][0]),
            float(game.state['price_ice'][0]),
            float(game.state['cups']),
            float(game.state['lemons']),
            float(game.state['sugar']),
            float(game.state['ice']),
            float(game.state['money']),
            float(game.state['popularity'])
        ])
        prediction = best_model.predict(np.array([state]), verbose=0)[0]
        
        cups_to_buy = max(0, int(prediction[0] ))  # 0 to 50 cups (1250 max)
        lemons_to_buy = max(0, int(prediction[1]))  # 0 to 20 lemons (500 max)
        sugar_to_buy = max(0, int(prediction[2] ))  # 0 to 20 sugar (500 max)
        ice_to_buy = max(0, int(prediction[3] )) 
        
        # Ensure we don't overspend
        total_cost = (cups_to_buy * game.state['price_cups'][0] +
                        lemons_to_buy * game.state['price_lemons'][0] +
                        sugar_to_buy * game.state['price_sugar'][0] +
                        ice_to_buy * game.state['price_ice'][0])
        
        if total_cost > game.state['money']:
            scale_factor = game.state['money'] / total_cost
            cups_to_buy = int(cups_to_buy * scale_factor)
            lemons_to_buy = int(lemons_to_buy * scale_factor)
            sugar_to_buy = int(sugar_to_buy * scale_factor)
            ice_to_buy = int(ice_to_buy * scale_factor)
        
        print(f"Buying: {cups_to_buy} cups, {lemons_to_buy} lemons, {sugar_to_buy} sugar, {ice_to_buy} ice")
        
        game.buy_supplies('cups', cups_to_buy, 0)
        game.buy_supplies('lemons', lemons_to_buy, 0)
        game.buy_supplies('sugar', sugar_to_buy, 0)
        game.buy_supplies('ice', ice_to_buy, 0)
        recipe_amount1 = max(1, int(prediction[5]))
        recipe_amount2 = max(1, int(prediction[6] ))
        recipe_amount3 = max(1, int(prediction[7] ))
        game.set_recipe(recipe_amount1, recipe_amount2, recipe_amount3)
        
        results = game.simulate_day()
        daily_profit = results['total_sold'] * game.state['price'] - game.state['total_expenses']
        total_profit += daily_profit
        
        print(f"Day {game.state['day']}: Profit = {format_money(daily_profit)}")

    print(f"\nTotal profit over 30 days: {format_money(total_profit)}")
    print("Simulation complete.")

    print("\nBest model's decision-making:")
    test_state = np.array([
        25.0,  # temperature (25°C, a warm day)
        85,    # price_cups[0] (85 cents for 25 cups)
        75,    # price_lemons[0] (75 cents for 25 lemons)
        60,    # price_sugar[0] (60 cents for 25 cups of sugar)
        85,    # price_ice[0] (85 cents for 25 ice cubes)
        100,   # cups (current inventory)
        50,    # lemons (current inventory)
        75,    # sugar (current inventory)
        200,   # ice (current inventory)
        10000, # money (10000 cents = $100.00)
        50     # popularity (moderate popularity)
    ])
    prediction = best_model.predict(np.array([test_state]), verbose=0)[0]
    
    print(f"For a typical summer day (25°C) with average market prices and inventory:")
    print(f"  Cups to buy: {max(0, int(prediction[0] * 50)) }")
    print(f"  Lemons to buy: {max(0, int(prediction[1] * 20)) }")
    print(f"  Sugar to buy: {max(0, int(prediction[2] * 20)) }")
    print(f"  Ice to buy: {max(0, int(prediction[3] * 50)) }")
    print(f"  Price set: {format_money(max(1, min(prediction[4] * 2, 200)))}")
    recipe_amount1 = max(1, int(prediction[5]))
    recipe_amount2 = max(1, int(prediction[6]))
    recipe_amount3 = max(1, int(prediction[7] ))
    print(f"  Recipe: {recipe_amount1} lemons, {recipe_amount2} sugar, {recipe_amount3} ice")

