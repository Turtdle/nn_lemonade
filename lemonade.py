import random

class Customer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.direction = random.choice([-1, 1])
        self.x = -52 if self.direction == 1 else 623
        self.y = 160
        self.bought = False
        self.speed = 25
        self.bubble = -1
        self.bubble_time = 0

    def move(self):
        self.x += self.direction * self.speed
        if (self.direction == -1 and self.x <= -60) or (self.direction == 1 and self.x >= 572 + 60):
            return False
        return True

    def check_buy(self, state, weather_index):
        random1 = random.random()
        
        if (not self.bought):
            if (173 <= self.x <= 398):
                #print(f"buy_or_pass: {self.simulation.buy_or_pass(state, weather_index)}, random1: {random1}")
                if (self.simulation.buy_or_pass(state, weather_index) > random1):
                    #print("Buying!")
                    self.simulation.buy_glass(state, self)
                
                self.bought = True 

        return self.bought

    def add_bubble(self, bubble):
        self.bubble = bubble
        self.bubble_time = 15

    def update_bubble(self):
        if self.bubble_time >= 0:
            self.bubble_time -= 1

WEATHER = ['sunny', 'hazy', 'cloudy', 'overcast', 'rain']
def buy_ingredients(state):
    if state['buy_order'] == []:
        return
    total_cost = (state['buy_order'][0] * 3.48 + state['buy_order'][1] * 7.5 + state['buy_order'][2] * 7.75 + state['buy_order'][3] * 0.875)
    if total_cost <= state['money']:
        state['money'] -= total_cost
        state['cups'] += state['buy_order'][0]
        state['lemons'] += state['buy_order'][1]
        state['sugar'] += state['buy_order'][2]
        state['ice'] += state['buy_order'][3]
        state['buy_order'] = []
        state['failed_to_buy'] = False
        #print("Bought ingredients!")
    else:
        state['failed_to_buy'] = True
        #print("Failed to buy ingredients")
def simulate_day(state):
    day_state = state
    buy_ingredients(day_state)
    simulation = Simulation(day_state)
    #buy_ingredients(simulation.state)
    for _ in range(1000):
        simulation.update()
        if simulation.sold_out:
           # print("Sold out!")
            break


    state['total_income'] += simulation.total_sold * state['price']
    state['total_sold'] = simulation.total_sold
    state['total_customers'] = simulation.total_customers

    return state

class Simulation:
    def __init__(self, state):
        self.state = state
        self.total_sold = 0
        self.total_customers = 0
        self.in_pitcher = 0
        self.sold_out = False
        self.customers = []
        self.refill_pitcher()

    def update(self):
        self.add_customer()
        self.update_customers()
        self.check_sold_out()

    def add_customer(self):
        if random.random() < 0.1:
            self.customers.append(Customer(self))
            self.total_customers += 1

    def update_customers(self):
        weather_index = WEATHER.index(self.state['weather'])
        for customer in self.customers:
            customer.update_bubble()
            if not customer.move():
                self.customers.remove(customer)
            elif customer.check_buy(self.state, weather_index):
                self.give_rep(self.state)
                #customer.bought = False

    def check_sold_out(self):
        if self.in_pitcher == 0 and (self.state['cups'] == 0 or self.state['ice'] < self.state['recipe_ice'] or self.state['lemons'] < self.state['recipe_lemons'] or self.state['sugar'] < self.state['recipe_sugar']):
            self.sold_out = True

    def buy_or_pass(self, state, weather_index):
        demand = ((state['temperature'] - 10) / 40 + (5 - weather_index) / 20) * \
                 (((state['temperature'] / 4) - state['price']) / (state['temperature'] / 4) + 1)

        if state['rep_level'] < random.random() * (state['rep_level'] - 500):
            demand *= state['reputation']
        demand *= (state['recipe_lemons'] + 1) / 5
        demand *= (state['recipe_sugar'] + 4) / 8

        for customer in self.customers:
            if customer.bubble_time > 0:
                demand *= 1.3 if customer.bubble == 0 else 0.5
        #print(demand * 1.3)
        return (demand) * 1.3

    def buy_glass(self, state, customer):
        if not self.sold_out and self.in_pitcher > 0 and  state['cups'] > 0 and state['ice'] >= state['recipe_ice']:
            self.in_pitcher -= 1
            state['ice'] -= state['recipe_ice']
            state['cups'] -= 1
            state['money'] += state['price']
            self.total_sold += 1
            if self.in_pitcher == 0:
                self.refill_pitcher()
            if self.give_rep(state) < 1:
                bubble = self.check_bubble(state)
                if bubble > 0:
                    customer.add_bubble(bubble)
                elif random.random() < 0.3:
                    customer.bubble = 0
            #print("buy_glass is returning True")
            return True
        else:
            self.sold_out = True
            #print("buy_glass is returning False")
            return False


    def give_rep(self, state):
        opinion = 0.8 + random.random() * 0.4
        opinion *= state['recipe_lemons'] / 4
        opinion *= state['recipe_sugar'] / 4
        opinion *= state['recipe_ice'] / ((state['temperature'] - 10) / 5) + 1
        opinion *= ((state['temperature'] - 10) / 5 + 1) / (state['recipe_ice'] + 4)
        opinion *= (state['temperature'] / 4 - state['price']) / (state['temperature'] / 4) + 1
        opinion = min(max(opinion, 0), 2)
        state['reputation'] += opinion
        state['rep_level'] += 1
        return opinion

    def check_bubble(self, state):
        reasons = [0, 0, 0]

        if state['recipe_lemons'] < 4 or state['recipe_sugar'] < 4:
            reasons[2] = 1
            #print("Lemons and sugar too low")
            return 3
        if state['recipe_ice'] < (state['temperature'] - 49) / 5:
            reasons[1] = 1
            #print("Ice too low")
            return 2
        if state['price'] > state['temperature'] / 4:
            reasons[0] = 1
            #print("Price too high")
            return 1

        a = random.randint(0, 2)
        return a + 1 if reasons[a] == 1 else 0

    def refill_pitcher(self):
        state = self.state
        if self.in_pitcher == 0 and state['lemons'] >= state['recipe_lemons'] and state['sugar'] >= state['recipe_sugar']:
            self.in_pitcher = 8 + state['recipe_ice']
            state['lemons'] -= state['recipe_lemons']
            state['sugar'] -= state['recipe_sugar']
        else:
            self.sold_out = True
            #print("Out of lemonade!")

    

def main():
    # Example usage
    cur_state = {
        'cups': 107,
        'lemons': 57,
        'sugar': 32,
        'ice': 527,
        'money': 0,
        'temperature': 80,
        'weather': 'sunny',
        'price': 33,
        'recipe_lemons': 7,
        'recipe_sugar': 4,
        'recipe_ice': 5,
        'total_income': 0,
        'rep_level': 0,
        'reputation': 0,
        'buy_order' : [1,1,1,1],
        'failed_to_buy' : False
    }




    new_state = simulate_day(cur_state)
    print(f"Total glasses sold: {new_state['total_sold']} / {new_state['total_customers']} customers")
    print(f"Money earned: ${new_state['total_income']/100:.2f}")

if __name__ == '__main__':
    main()