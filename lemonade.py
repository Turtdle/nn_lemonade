import random

class LemonadeStand:
    def __init__(self, duration):
        self.state = {
            'duration': duration,
            'day': 0,
            'cups': 0,
            'lemons': 0,
            'sugar': 0,
            'ice': 0,
            'money': 2000,  # Cents
            'temperature': 0,
            'temperature_farenheit': 0,
            'weather': None,
            'price': 25,  # Cents
            'recipe_lemons': 4,
            'recipe_sugar': 4,
            'recipe_ice': 4,
            'popularity': 0,
            'total_income': 0,
            'total_expenses': 0,
            'price_cups': [0, 0, 0],
            'price_lemons': [0, 0, 0],
            'price_sugar': [0, 0, 0],
            'price_ice': [0, 0, 0],
            'rep_level': 0,
            'reputation': 0
        }
        self.weather_options = ['sunny', 'hazy', 'cloudy', 'overcast', 'rain']
        self.weather_names = {
            'sunny': 'Clear and Sunny',
            'hazy': 'Hazy',
            'cloudy': 'Cloudy',
            'overcast': 'Overcast',
            'rain': 'Rain!'
        }
        self.new_day()

    def new_day(self):
        self.state['day'] += 1
        self.state['weather'] = random.choice(self.weather_options)
        self.state['temperature'] = random.randint(11, 37)
        self.state['temperature_farenheit'] = round(self.state['temperature'] * 9/5 + 32)

        self.state['price_cups'] = [
            random.randint(75, 100),
            random.randint(150, 175),
            random.randint(275, 325)
        ]
        self.state['price_lemons'] = [
            random.randint(50, 100),
            random.randint(200, 250),
            random.randint(400, 450)
        ]
        self.state['price_sugar'] = [
            random.randint(50, 75),
            random.randint(150, 175),
            random.randint(325, 350)
        ]
        self.state['price_ice'] = [
            random.randint(75, 100),
            random.randint(200, 225),
            random.randint(350, 400)
        ]

    def buy_supplies(self, item, quantity, price_index):
        price = self.state[f'price_{item}'][price_index]
        total_cost = price * quantity
        if total_cost > self.state['money']:
            return False
        self.state[item] += quantity
        self.state['money'] -= total_cost
        self.state['total_expenses'] += total_cost
        return True

    def set_price(self, price):
        self.state['price'] = price

    def set_recipe(self, lemons, sugar, ice):
        self.state['recipe_lemons'] = lemons
        self.state['recipe_sugar'] = sugar
        self.state['recipe_ice'] = ice

    def simulate_day(self):
        pitcher = 0
        total_sold = 0
        total_customers = 0
        sold_out = False

        def refill_pitcher():
            nonlocal pitcher, sold_out
            if pitcher == 0 and self.state['lemons'] >= self.state['recipe_lemons'] and self.state['sugar'] >= self.state['recipe_sugar']:
                pitcher = 8 + self.state['recipe_ice']
                self.state['lemons'] -= self.state['recipe_lemons']
                self.state['sugar'] -= self.state['recipe_sugar']
            
            if pitcher == 0 or self.state['cups'] == 0 or self.state['ice'] < self.state['recipe_ice']:
                sold_out = True

        def buy_or_pass():
            weather_index = self.weather_options.index(self.state['weather'])
            demand = ((self.state['temperature_farenheit'] - 50) / 200 + (5 - weather_index) / 20) * \
                     (((self.state['temperature_farenheit'] / 4) - self.state['price']) / (self.state['temperature_farenheit'] / 4) + 1)
            if self.state['rep_level'] < random.random() * (self.state['rep_level'] - 500):
                demand = demand * self.state['reputation']
            demand *= (self.state['recipe_lemons'] + 1) / 5
            demand *= (self.state['recipe_sugar'] + 4) / 8
            return (demand + random.uniform(-0.1, 0.1)) * 1.3 > random.random()

        def sell_glass():
            nonlocal pitcher, total_sold
            if not sold_out and pitcher > 0 and self.state['cups'] > 0 and self.state['ice'] >= self.state['recipe_ice']:
                pitcher -= 1
                self.state['ice'] -= self.state['recipe_ice']
                self.state['cups'] -= 1
                self.state['money'] += self.state['price']
                self.state['total_income'] += self.state['price']
                total_sold += 1
                refill_pitcher()
                return True
            return False

        for _ in range(100):  # Simulate 100 potential customers
            if random.random() < 0.1:  # 10% chance of customer appearing
                total_customers += 1
                if buy_or_pass():
                    sell_glass()

            if sold_out:
                break

        self.state['reputation'] += total_sold * 0.5  # Simple reputation increase
        self.state['rep_level'] += total_customers

        return {
            'total_sold': total_sold,
            'total_customers': total_customers,
            'sold_out': sold_out
        }

    def end_season(self):
        inventory_value = self.state['cups'] * 2 + self.state['lemons'] * 4 + self.state['sugar'] * 6
        outcome = inventory_value + self.state['total_income'] - self.state['total_expenses']
        return {
            'inventory_value': inventory_value,
            'outcome': outcome
        }

# Helper function to format money
def format_money(cents):
    dollars = int(cents) // 100
    cents = int(cents) % 100
    return f"${dollars}.{cents:02d}"


game = LemonadeStand(duration=7)  # 7-day game

for _ in range(game.state['duration']):
    game.new_day()
    # Here you would add code to display the day's info and get player input
    game.buy_supplies('cups', 100, 0)  # Buy 10 cups at the lowest price
    game.set_price(30)  # Set lemonade price to 30 cents
    game.set_recipe(4, 4, 5)  # 4 lemons, 4 sugar, 5 ice
    results = game.simulate_day()
    # Here you would display the results to the player

final_results = game.end_season()
print(f"Total income: {format_money(final_results['outcome'])}")
print(f"Inventory value: {format_money(final_results['inventory_value'])}")

