import random
from typing import Dict, Tuple

class LemonadeStand:
    def __init__(self, prices: Dict[str, float], money: float, temperature: float, weather_index: int):
        self.prices = prices
        self.state = {
            "temperature": temperature,
            "weather_index": weather_index,
            "reputation": 500,
            "money": money,
            "total_income": 0,
            "cups": 0,
            "lemons": 0,
            "sugar": 0,
            "ice": 0
        }

    def simulate_day(self, recipe: Dict[str, int], sale_price: float) -> float:
        customers = random.randint(50, 100)
        profit = 0
        sales = 0

        for _ in range(customers):
            if self.state["cups"] > 0 and self.state["lemons"] >= recipe["lemons"] and \
               self.state["sugar"] >= recipe["sugar"] and self.state["ice"] >= recipe["ice"]:
                
                demand = self.calculate_demand(recipe, sale_price)
                if random.random() < demand:
                    self.state["cups"] -= 1
                    self.state["lemons"] -= recipe["lemons"]
                    self.state["sugar"] -= recipe["sugar"]
                    self.state["ice"] -= recipe["ice"]
                    self.state["money"] += sale_price
                    self.state["total_income"] += sale_price
                    profit += sale_price
                    sales += 1

        print(f"Debug: Customers: {customers}, Sales: {sales}, Profit: ${profit:.2f}")
        return profit

    def calculate_demand(self, recipe: Dict[str, int], price: float) -> float:
        temperature_factor = (self.state["temperature"] - 50) / 200
        weather_factor = (5 - self.state["weather_index"]) / 20
        price_factor = ((self.state["temperature"] / 4 - price) / (self.state["temperature"] / 4) + 1)
        
        base_demand = temperature_factor + weather_factor
        base_demand *= price_factor
        
        if self.state["reputation"] < random.random() * (self.state["reputation"] - 500):
            base_demand *= self.state["reputation"] / 1000

        recipe_factor = ((recipe["lemons"] + 1) / 5) * ((recipe["sugar"] + 4) / 8)
        demand = base_demand * recipe_factor

        # Add some randomness
        demand += random.uniform(-0.1, 0.1)
        
        demand = min(max(demand, 0), 1)
        print(f"Debug: Demand: {demand:.4f}")
        return demand

    def buy_ingredients(self, quantities: Dict[str, int]) -> bool:
        total_cost = sum(quantities[item] * self.prices[item] for item in quantities)
        if total_cost <= self.state["money"]:
            for item, quantity in quantities.items():
                self.state[item] += quantity
            self.state["money"] -= total_cost
            return True
        return False

def optimize_stand(stand: LemonadeStand) -> Tuple[Dict[str, int], float, Dict[str, int]]:
    best_recipe = {"lemons": 4, "sugar": 4, "ice": 4}
    best_price = 0.50
    best_purchase = {"cups": 50, "lemons": 200, "sugar": 200, "ice": 200}
    best_profit = float('-inf')

    for _ in range(10000):  # Increased number of iterations for better optimization
        recipe = {
            "lemons": random.randint(1, 8),
            "sugar": random.randint(1, 8),
            "ice": random.randint(1, 8)
        }
        sale_price = random.uniform(0.10, 0.99)
        
        # Determine purchase quantities
        max_sales = 100  # Maximum possible sales in a day
        purchase = {
            "cups": random.randint(50, max_sales),
            "lemons": random.randint(50, max_sales * 2),
            "sugar": random.randint(50, max_sales * 2),
            "ice": random.randint(50, max_sales * 2)
        }

        # Create a copy of the stand for this iteration
        test_stand = LemonadeStand(stand.prices, stand.state["money"], stand.state["temperature"], stand.state["weather_index"])

        if not test_stand.buy_ingredients(purchase):
            continue  # Skip this iteration if we can't afford the ingredients

        profit = test_stand.simulate_day(recipe, sale_price)

        if profit > best_profit:
            best_profit = profit
            best_recipe = recipe
            best_price = sale_price
            best_purchase = purchase

    return best_recipe, best_price, best_purchase

# Get input from user
prices = {}
for item in ["cups", "lemons", "sugar", "ice"]:
    while True:
        try:
            price = float(input(f"Enter the price for one {item} (in cents): ")) / 100
            if price <= 0:
                raise ValueError
            prices[item] = price
            break
        except ValueError:
            print("Please enter a valid positive number.")

while True:
    try:
        money = float(input("Enter the amount of money you have (in dollars): "))
        if money <= 0:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid positive number.")

while True:
    try:
        temperature = float(input("Enter the temperature (in Fahrenheit): "))
        if temperature < 0 or temperature > 120:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid temperature between 0 and 120.")

while True:
    try:
        weather_index = int(input("Enter the weather index (1-5, where 1 is worst and 5 is best): "))
        if weather_index < 1 or weather_index > 5:
            raise ValueError
        break
    except ValueError:
        print("Please enter a valid weather index between 1 and 5.")

# Create stand and run optimization
stand = LemonadeStand(prices, money, temperature, weather_index)
optimal_recipe, optimal_price, optimal_purchase = optimize_stand(stand)

print("\nOptimal Strategy:")
print(f"Recipe:")
print(f"  Lemons: {optimal_recipe['lemons']}")
print(f"  Sugar: {optimal_recipe['sugar']}")
print(f"  Ice: {optimal_recipe['ice']}")
print(f"Sale Price: {optimal_price:.2f}¢")
print("\nOptimal Purchase Quantities:")
for item, quantity in optimal_purchase.items():
    print(f"  {item.capitalize()}: {quantity}")
print("\nCosts and Profit:")
total_cost = sum(optimal_purchase[item] * prices[item] for item in optimal_purchase)
print(f"  Total Cost of Ingredients: ${total_cost:.2f}")
print(f"  Remaining Money: ${stand.state['money'] - total_cost:.2f}")
print(f"  Estimated Profit: ${stand.simulate_day(optimal_recipe, optimal_price):.2f}")
# After optimization, simulate the day with the best strategy for debugging
print("\nSimulating the day with the optimal strategy:")
stand.buy_ingredients(optimal_purchase)
final_profit = stand.simulate_day(optimal_recipe, optimal_price)

print(f"\nFinal Profit: ${final_profit:.2f}")
print(f"Remaining Ingredients:")
print(f"  Cups: {stand.state['cups']}")
print(f"  Lemons: {stand.state['lemons']}")
print(f"  Sugar: {stand.state['sugar']}")
print(f"  Ice: {stand.state['ice']}")