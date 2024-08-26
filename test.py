from lemonade import LemonadeStand, format_money

game = LemonadeStand(duration=7)
game.set_price(5)
game.set_recipe(4, 5, 5)

game = LemonadeStand(duration=7)
game.set_price(50)  # Set price to 25 cents
game.set_recipe(8, 8, 8)  # Set recipe to 4 lemons, 4 sugar, 4 ice

for day in range(game.state['duration']):
    game.new_day()
    print(f"\nDay {day + 1}:")
    print(f"Weather: {game.state['weather']}, Temperature: {game.state['temperature']}°C ({game.state['temperature_farenheit']}°F)")
    
    print("\nBefore buying supplies:")
    print(f"Money: {format_money(game.state['money'])}")
    print(f"Cups: {game.state['cups']}")
    print(f"Lemons: {game.state['lemons']}")
    print(f"Sugar: {game.state['sugar']}")
    print(f"Ice: {game.state['ice']}")
    
    game.buy_supplies('cups', 5, 0)
    game.buy_supplies('lemons', 5, 0)
    game.buy_supplies('sugar', 5, 0)
    game.buy_supplies('ice', 5, 0)
    
    print("\nAfter buying supplies:")
    print(f"Money: {format_money(game.state['money'])}")
    print(f"Cups: {game.state['cups']}")
    print(f"Lemons: {game.state['lemons']}")
    print(f"Sugar: {game.state['sugar']}")
    print(f"Ice: {game.state['ice']}")
    
    results = game.simulate_day()
    print(f"Sold: {results['total_sold']} cups")
    print(f"Customers: {results['total_customers']}")
    print(f"Sold out: {results['sold_out']}")
    print(f"Money: {format_money(game.state['money'])}")

final_results = game.end_season()
print(f"\nFinal Results:")
print(f"Total income: {format_money(final_results['outcome'])}")
print(f"Inventory value: {format_money(final_results['inventory_value'])}")