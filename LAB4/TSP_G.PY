import random
import math
import matplotlib.pyplot as plt

# Function to calculate the total length of a tour
def calculate_total_length(route, distance_matrix):
    total_length = 0
    for i in range(len(route)):
        total_length += distance_matrix[route[i - 1]][route[i]]
    return total_length

# Function to generate a random initial route
def generate_initial_route(city_count):
    route = list(range(city_count))
    random.shuffle(route)
    return route

# Function to create a neighboring route by reversing a segment
def generate_neighboring_route(route):
    i, j = sorted(random.sample(range(len(route)), 2))
    route[i:j + 1] = reversed(route[i:j + 1])
    return route

# Simulated annealing algorithm for the Traveling Salesman Problem
def simulated_annealing_algorithm(distance_matrix, num_iterations, start_temperature, temp_decay_rate):
    city_count = len(distance_matrix)
    current_route = generate_initial_route(city_count)
    current_length = calculate_total_length(current_route, distance_matrix)
    optimal_route = current_route[:]
    optimal_length = current_length

    temperature = start_temperature

    for iteration in range(num_iterations):
        new_route = generate_neighboring_route(current_route)
        new_length = calculate_total_length(new_route, distance_matrix)

        # Determine whether to accept the new route
        if new_length < current_length or random.random() < math.exp(
                (current_length - new_length) / temperature):
            current_route = new_route
            current_length = new_length

            # Update the best route if needed
            if current_length < optimal_length:
                optimal_route = current_route[:]
                optimal_length = current_length

        # Reduce the temperature
        temperature *= temp_decay_rate

    return optimal_route, optimal_length


# Example distance matrix (for 4 cities)
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

num_iterations = 10000
start_temperature = 100.0
temp_decay_rate = 0.90

# Apply simulated annealing to find the best tour
optimal_route, optimal_length = simulated_annealing_algorithm(
    distance_matrix, num_iterations, start_temperature, temp_decay_rate)

# Visualization of the best route
city_count = len(distance_matrix)
x_coords = [random.random() * 100 for _ in range(city_count)]
y_coords = [random.random() * 100 for _ in range(city_count)]

plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, c='blue', s=100)
for i, city in enumerate(range(city_count)):
    plt.annotate(city, (x_coords[i], y_coords[i]), fontsize=12)

for i in range(city_count - 1):
    plt.plot([x_coords[optimal_route[i]], x_coords[optimal_route[i + 1]]],
             [y_coords[optimal_route[i]], y_coords[optimal_route[i + 1]]], c='red')
plt.plot([x_coords[optimal_route[-1]], x_coords[optimal_route[0]]],
         [y_coords[optimal_route[-1]], y_coords[optimal_route[0]]], c='red')

plt.title("Traveling Salesman Problem - Optimal Route")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.grid(True)
plt.show()

print("Optimal Route:", optimal_route)
print("Optimal Distance:", optimal_length)
