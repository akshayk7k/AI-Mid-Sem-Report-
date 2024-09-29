import random
import math
import matplotlib.pyplot as plt

rajasthan_tourist_spots = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Pushkar": (26.4897, 74.5511),
    "Jaisalmer": (26.9157, 70.9083),
    "Ajmer": (26.4499, 74.6399),
    "Mount Abu": (24.5925, 72.7156),
    "Bikaner": (28.0229, 73.3119),
    "Ranthambore": (25.8667, 76.3),
    "Chittorgarh": (24.8887, 74.6269),
    "Bundi": (25.4415, 75.6454),
    "Alwar": (27.5530, 76.6346),
    "Bharatpur": (27.1767, 77.6844),
    "Kota": (25.2138, 75.8648),
    "Sawai Madhopur": (25.9928, 76.3526),
    "Shekhawati": (27.7366, 75.9730),
    "Dungarpur": (23.8363, 73.7143),
    "Nathdwara": (24.9339, 73.8226),
    "Mandawa": (28.0556, 75.1419),
    "Osian": (26.9112, 72.3917)
}

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    earth_radius = 6371
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(delta_lon / 2) ** 2
    central_angle = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * central_angle

def compute_total_distance(tour):
    total_distance = 0
    for i in range(len(tour)):
        total_distance += calculate_distance(rajasthan_tourist_spots[tour[i]],
                                             rajasthan_tourist_spots[tour[(i + 1) % len(tour)]])
    return total_distance

def simulated_annealing_optimizer(locations, initial_temp=1000, cooling_rate=0.99, max_iterations=100000):
    current_route = list(locations.keys())
    random.shuffle(current_route)
    current_cost = compute_total_distance(current_route)
    best_route = current_route[:]
    best_cost = current_cost

    temperature = initial_temp
    cost_history = []

    for _ in range(max_iterations):
        new_route = current_route[:]
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_cost = compute_total_distance(new_route)

        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_route = new_route[:]
            current_cost = new_cost

            if new_cost < best_cost:
                best_route = new_route[:]
                best_cost = new_cost

        cost_history.append(current_cost)
        temperature *= cooling_rate

    best_route.append(best_route[0])

    return best_route, best_cost, cost_history

optimal_tour, optimal_cost, cost_progression = simulated_annealing_optimizer(rajasthan_tourist_spots)

intermediate_distances = []
for i in range(len(optimal_tour) - 1):
    city1 = optimal_tour[i]
    city2 = optimal_tour[i + 1]
    intermediate_distances.append(calculate_distance(rajasthan_tourist_spots[city1], rajasthan_tourist_spots[city2]))

print("Optimal tour:", optimal_tour)
print("Optimal cost:", optimal_cost, "km")
print("Intermediate distances:", intermediate_distances)

plt.figure(figsize=(10, 8))
for city, (latitude, longitude) in rajasthan_tourist_spots.items():
    plt.plot(longitude, latitude, 'o', markersize=8, label=city)
plt.plot([rajasthan_tourist_spots[optimal_tour[i]][1] for i in range(len(optimal_tour))],
         [rajasthan_tourist_spots[optimal_tour[i]][0] for i in range(len(optimal_tour))],
         'r-')
plt.title('Optimal Tour in Rajasthan')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
