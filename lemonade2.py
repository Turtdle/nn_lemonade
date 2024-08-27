import random
import time

class Simulation:
    def __init__(self):
        self.images = None
        self.context = None
        self.time = 0
        self.duration = 1000
        self.frame_rate = 4

        self.on_update = None
        self.on_step = None
        self.on_stop = None

        self._request_id = -1
        self._last_time = -1
        self._frame_time = 0
        self.stopped = True

        self.customers = []
        self.in_pitcher = 0
        self.sold_out = False
        self.total_sold = 0
        self.total_customers = 0

    def start(self):
        self.stopped = False
        self.time = 0
        self._frame_time = 0
        self.in_pitcher = 0
        self.sold_out = False
        self.total_sold = 0
        self.total_customers = 0

        Customer.reset()

        self._last_time = time.time()

        self.refill_pitcher()
        self.step()

    def refill_pitcher(self):
        if (self.in_pitcher == 0 and
            state.data['lemons'] >= state.data['recipe_lemons'] and
            state.data['sugar'] >= state.data['recipe_sugar']):
            self.in_pitcher = 8 + state.data['recipe_ice']
            state.data['lemons'] -= state.data['recipe_lemons']
            state.data['sugar'] -= state.data['recipe_sugar']

        if self.in_pitcher == 0 or state.data['cups'] == 0 or state.data['ice'] < state.data['recipe_ice']:
            self.sold_out = True

    def buy_or_pass(self):
        demand = ((state.data['temperature_farenheit'] - 50) / 200 + (5 - self.weather_index) / 20) * \
                 (((state.data['temperature_farenheit'] / 4) - state.data['price']) / (state.data['temperature_farenheit'] / 4) + 1)
        if state.data['rep_level'] < random.random() * (state.data['rep_level'] - 500):
            demand = demand * state.data['reputation']
        demand *= (state.data['recipe_lemons'] + 1) / 5
        demand *= (state.data['recipe_sugar'] + 4) / 8

        for customer in self.customers:
            if customer.bubble_time > 0:
                demand *= 1.3 if customer.bubble == 0 else 0.5

        return (demand + random.uniform(-0.1, 0.1)) * 1.3

    def buy_glass(self, customer):
        if (not self.sold_out and self.in_pitcher > 0 and
            state.data['cups'] > 0 and state.data['ice'] >= state.data['recipe_ice']):
            self.in_pitcher -= 1
            state.data['ice'] -= state.data['recipe_ice']
            state.data['cups'] -= 1
            state.data['money'] += state.data['price']
            state.data['total_income'] += state.data['price']
            self.total_sold += 1

            self.refill_pitcher()

            if self.give_rep() < 1:
                bubble = self.check_bubble()
                if bubble > 0:
                    customer.add_bubble(bubble)
            elif random.random() < 0.3:
                customer.add_bubble(0)

            return True
        else:
            self.sold_out = True
            return False

    def give_rep(self):
        opinion = 0.8 + random.random() * 0.4
        opinion *= state.data['recipe_lemons'] / 4
        opinion *= state.data['recipe_sugar'] / 4
        opinion *= state.data['recipe_ice'] / ((state.data['temperature_farenheit'] - 50) / 5) + 1
        opinion *= ((state.data['temperature_farenheit'] - 50) / 5 + 1) / (state.data['recipe_ice'] + 4)
        opinion *= (state.data['temperature_farenheit'] / 4 - state.data['price']) / (state.data['temperature_farenheit'] / 4) + 1
        opinion = min(max(opinion, 0), 2)
        state.data['reputation'] += opinion
        state.data['rep_level'] += 1
        return opinion

    def check_bubble(self):
        # Yuck!
        if state.data['recipe_lemons'] < 4 or state.data['recipe_sugar'] < 4:
            _reasons[2] = 1
        else:
            _reasons[2] = 0

        # More ice!
        if state.data['recipe_ice'] < (state.data['temperature_farenheit'] - 49) / 5:
            _reasons[1] = 1
        else:
            _reasons[1] = 0

        # $$!
        if state.data['price'] > state.data['temperature_farenheit'] / 4:
            _reasons[0] = 1
        else:
            _reasons[0] = 0

        a = random.randint(0, 2)
        return a + 1 if _reasons[a] == 1 else 0

    def draw_rain(self):
        # This method would need to be implemented differently in Python,
        # as it depends on a graphics context that's not defined here.
        pass

    def update(self, now):
        if self.stopped:
            return

        elapsed = min(0.1, (now - self._last_time) / 1000)
        self._last_time = now
        self._frame_time += elapsed

        if self.on_update:
            self.on_update(self)

        if self._frame_time >= 1 / self.frame_rate:
            self.step()
            self._frame_time -= 1 / self.frame_rate

        if self.time >= self.duration:
            self.stop()
        else:
            # In Python, you'd need to use a different method for animation frames
            # This is just a placeholder
            self._request_id = self.schedule_next_update()

    def step(self):
        self.time += 1

        if random.random() < 0.1:
            self.add_customer()

        # The drawing methods would need to be implemented differently in Python
        # as they depend on a graphics context that's not defined here.

        if state.data['weather'] == 'rain':
            self.draw_rain()

        if self.on_step:
            self.on_step(self)

    def add_customer(self):
        self.customers.append(Customer(self))
        self.total_customers += 1

    def stop(self):
        self.stopped = True
        self.customers.clear()
        if self._request_id != -1:
            # You'd need to implement a method to cancel the scheduled update
            self.cancel_scheduled_update(self._request_id)
            self._request_id = -1

        if self.on_stop:
            self.on_stop(self)

    @property
    def weather_index(self):
        return WEATHER_TO_NUMBER[state.data['weather']]

    def schedule_next_update(self):
        # This method would need to be implemented to schedule the next update
        # It might use threading or an event loop, depending on your needs
        pass

    def cancel_scheduled_update(self, request_id):
        # This method would need to be implemented to cancel a scheduled update
        pass