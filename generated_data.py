import numpy as np
import csv

def generate_random_data(num_samples=10000):
    x_train = np.zeros((num_samples, 9))
    y_train = np.zeros((num_samples, 8))

    for i in range(num_samples):
        # Generate x_train data
        x_train[i] = [
            np.random.randint(0, 1000),  # amount of cups
            np.random.randint(0, 1000),  # amount of lemons
            np.random.randint(0, 1000),  # amount of sugar
            np.random.randint(0, 2000),  # amount of ice
            np.random.randint(1000, 10000),  # amount of money (in cents)
            np.random.randint(50, 100),  # temp
            np.random.randint(0, 5),  # weather (0-5)
            np.random.randint(0, 100),  # rep_level
            np.random.uniform(0.5, 1.5)  # reputation
        ]

        # Generate y_train data
        y_train[i] = [
            np.random.randint(0, 200),  # amount of cups to buy
            np.random.randint(0, 100),  # amount of lemons to buy
            np.random.randint(0, 100),  # amount of sugar to buy
            np.random.randint(0, 500),  # amount of ice to buy
            np.random.uniform(0.25, 2.00),  # price
            np.random.uniform(0.1, 1.0),  # recipe_lemons
            np.random.uniform(0.1, 1.0),  # recipe_sugar
            np.random.uniform(0.1, 1.0)  # recipe_ice
        ]

    return x_train, y_train

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
    x_train, y_train = generate_random_data(10000)
    
    save_to_csv(x_train, 'x_train_random.csv')
    save_to_csv(y_train, 'y_train_random.csv')
    
    print("Random training data generated and saved to x_train_random.csv and y_train_random.csv")