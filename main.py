import numpy as np
from keras.models import clone_model
from lemonade import LemonadeStand, format_money
from neural_net import build_model
import time

class EvolutionaryTrainer:
    def __init__(self, population_size=50, generations=5, mutation_rate=0.1):
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
            mask = np.random.random(size=w1.shape) > 0.5
            new_w = np.where(mask, w1, w2)
            new_weights.append(new_w)
        new_model.set_weights(new_weights)
        return new_model

    def evaluate_model(self, model):
        game = LemonadeStand(duration=7)
        total_profit = 0
        for _ in range(game.state['duration']):
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
            prediction = model.predict(np.array([state]), verbose=0)[0]
            
            cups_to_buy = max(0, int(prediction[0] * 100))
            game.buy_supplies('cups', cups_to_buy, 0)
            game.buy_supplies('lemons', max(0, int(prediction[1] * 50)), 0)
            game.buy_supplies('sugar', max(0, int(prediction[2] * 50)), 0)
            game.buy_supplies('ice', max(0, int(prediction[3] * 100)), 0)
            game.set_price(max(1, int(prediction[4] * 100)))  # Ensure price is at least 1 cent
            recipe_amount = max(1, int(prediction[5] * 10))  # Ensure at least 1 of each ingredient
            game.set_recipe(recipe_amount, recipe_amount, recipe_amount)
            
            results = game.simulate_day()
            total_profit += results['total_sold'] * game.state['price'] - game.state['total_expenses']

        return total_profit

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
        
        cups_to_buy = max(0, int(prediction[0] * 100))
        game.buy_supplies('cups', cups_to_buy, 0)
        game.buy_supplies('lemons', max(0, int(prediction[1] * 50)), 0)
        game.buy_supplies('sugar', max(0, int(prediction[2] * 50)), 0)
        game.buy_supplies('ice', max(0, int(prediction[3] * 100)), 0)
        game.set_price(max(1, int(prediction[4] * 100)))  # Ensure price is at least 1 cent
        recipe_amount = max(1, int(prediction[5] * 10))  # Ensure at least 1 of each ingredient
        game.set_recipe(recipe_amount, recipe_amount, recipe_amount)
        
        results = game.simulate_day()
        daily_profit = results['total_sold'] * game.state['price'] - game.state['total_expenses']
        total_profit += daily_profit
        
        print(f"Day {game.state['day']}: Profit = {format_money(daily_profit)}")

    print(f"\nTotal profit over 30 days: {format_money(total_profit)}")
    print("Simulation complete.")

    print("\nBest model's decision-making:")
    test_state = np.array([
        25.0,  # temperature (25°C)
        0.37,  # price_cups
        0.08,  # price_lemons
        0.07,  # price_sugar
        0.008,  # price_ice
        0,  # cups
        0,  # lemons
        0,  # sugar
        0,  # ice
        20,  # money (50.00)
        0  # popularity
    ])
    prediction = best_model.predict(np.array([test_state]), verbose=0)[0]
    
    print(f"For a typical summer day (25°C) with average market prices and inventory:")
    print(f"  Cups to buy: {max(0, int(prediction[0] * 100))}")
    print(f"  Lemons to buy: {max(0, int(prediction[1] * 50))}")
    print(f"  Sugar to buy: {max(0, int(prediction[2] * 50))}")
    print(f"  Ice to buy: {max(0, int(prediction[3] * 100))}")
    print(f"  Price set: {format_money(max(1, int(prediction[4] * 100)))}")
    recipe_amount = max(1, int(prediction[5] * 10))
    print(f"  Recipe: {recipe_amount} lemons, {recipe_amount} sugar, {recipe_amount} ice")