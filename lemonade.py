import random

def buy_or_pass(temperature_farenheit, weather_index, price, rep_level, reputation, recipe_lemons, recipe_sugar, customers):
    demand = ((temperature_farenheit - 50) / 200 + (5 - weather_index) / 20) * \
             (((temperature_farenheit / 4) - price) / (temperature_farenheit / 4) + 1)
    
    if rep_level < random.random() * (rep_level - 500):
        demand = demand * reputation
    demand *= (recipe_lemons + 1) / 5
    demand *= (recipe_sugar + 4) / 8

    for customer in customers:
        if customer['bubble_time'] > 0:
            demand *= 1.3 if customer['bubble'] == 0 else 0.5

    return (demand + random.uniform(-0.1, 0.1)) * 1.3