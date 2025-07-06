import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Define the problem parameters
num_customers = 15
max_vehicles = 20
min_demand = 1
max_demand = 20
depot = (50, 50)

# Genetic Algorithm Parameters
population_size = 50
num_generations = 300
mutation_rate = 0.1
elite_percentage = 0.1

# Randomly generated customer locations and demands
customer_locations = np.random.rand(num_customers, 2) * 100
customer_demands = np.random.randint(min_demand, max_demand, size=num_customers)

# Define vehicle capacities
vehicle_capacities = [50] * max_vehicles

# Traffic Data - Simulated as random factors (1 means no traffic, >1 means more traffic)
traffic_factors = np.random.uniform(1.0, 2.0, (num_customers + 1, num_customers + 1))

# Distance calculation with traffic data
def calculate_distance(i, j):
    base_distance = np.linalg.norm(np.array(depot if i == 0 else customer_locations[i - 1]) -
                                   np.array(depot if j == 0 else customer_locations[j - 1]))
    return base_distance * traffic_factors[i][j]  # Apply traffic factor

# Distance matrix including traffic effects
distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
for i in range(num_customers + 1):
    for j in range(num_customers + 1):
        distance_matrix[i, j] = calculate_distance(i, j)

# Initialize population for genetic algorithm
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = random.sample(range(1, num_customers + 1), num_customers)
        population.append(individual)
    return population

# Fitness function: Ensure vehicle capacities are respected
def evaluate_fitness(chromosome):
    routes = [[] for _ in range(max_vehicles)]
    vehicle_capacities_remaining = list(vehicle_capacities)
    total_distance = 0

    # Assign customers to vehicles
    for customer in chromosome:
        assigned = False
        for idx, route in enumerate(routes):
            # Check if the customer can be assigned to this vehicle without exceeding capacity
            if vehicle_capacities_remaining[idx] >= customer_demands[customer - 1]:
                route.append(customer)
                vehicle_capacities_remaining[idx] -= customer_demands[customer - 1]
                assigned = True
                break

        # If we couldn't assign this customer, return a high penalty fitness value
        if not assigned:
            return float('inf'), routes  # Infinite fitness means an invalid solution

    # Calculate total distance, including traffic effects
    for route in routes:
        for i, j in zip(route, route[1:]):
            total_distance += distance_matrix[i][j]

    return 1 / total_distance, routes

def selection(population):
    tournament_size = 3
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(population, tournament_size)
        best_parent = max(tournament, key=lambda x: evaluate_fitness(x)[0])
        selected_parents.append(best_parent)
    return selected_parents

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, num_customers - 1)
    child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]

    # Check vehicle capacities and reassign customers if needed
    child1 = assign_customers_to_vehicles(child1)
    child2 = assign_customers_to_vehicles(child2)

    return child1, child2

# Mutation
def mutate(chromosome):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(num_customers), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    # Reassign customers to vehicles while respecting capacity constraints
    chromosome = assign_customers_to_vehicles(chromosome)
    return chromosome

# Helper function to reassign customers to vehicles, respecting capacity
def assign_customers_to_vehicles(chromosome):
    routes = [[] for _ in range(max_vehicles)]
    vehicle_capacities_remaining = list(vehicle_capacities)

    for customer in chromosome:
        assigned = False
        for idx, route in enumerate(routes):
            if vehicle_capacities_remaining[idx] >= customer_demands[customer - 1]:
                route.append(customer)
                vehicle_capacities_remaining[idx] -= customer_demands[customer - 1]
                assigned = True
                break

        if not assigned:
            raise ValueError("Cannot assign customer to any vehicle, capacity exceeded.")

    # Flatten the routes back to a single list of customers in order, preserving capacity constraints
    return [customer for route in routes for customer in route]
# Elitism (Retain the best solutions)
def elitism(population):
    elite_size = int(population_size * elite_percentage)
    elite = sorted(population, key=lambda x: evaluate_fitness(x)[0], reverse=True)[:elite_size]
    return elite

# Main Genetic Algorithm
def genetic_algorithm():
    start_time = time.time()
    best_fitnesses = []
    population = initialize_population()

    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = elitism(population) + new_population
        best_fitnesses.append(max([evaluate_fitness(chromosome)[0] for chromosome in population]))
        print(f"Generation {generation + 1}, Best Fitness: {best_fitnesses[generation]}")

    best_chromosome = max(population, key=lambda x: evaluate_fitness(x)[0])
    best_fitness, best_routes = evaluate_fitness(best_chromosome)
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time}s')

    # Calculate the total distance for the best route
    best_route_length = 0
    for route in best_routes:
        for i, j in zip(route, route[1:]):
            best_route_length += distance_matrix[i][j]

    return best_fitness, best_routes, best_route_length, best_fitnesses

def plot_routes(routes):
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink']
    plt.figure(figsize=(10, 8))

    # Plot each route with a unique color
    for idx, route in enumerate(routes):
        route_points = [depot] + [customer_locations[customer - 1] for customer in route] + [depot]
        plt.plot(
            [point[0] for point in route_points],
            [point[1] for point in route_points],
            color=colors[idx % len(colors)]
        )

        # Annotate route points with customer index and demand
        for i, point in enumerate(route_points):
            if i == 0 or i == len(route_points) - 1:  # Skip depot
                continue
            customer_idx = route[i - 1]
            plt.text(
                point[0] + 2, point[1],  # Slightly offset to the right
                f"{customer_idx}({customer_demands[customer_idx - 1]})",
                color='black', fontsize=10, ha='center'
            )

    # Plot customers and depot
    plt.scatter(customer_locations[:, 0], customer_locations[:, 1], color='blue', label='Customers')
    plt.scatter(depot[0], depot[1], color='black', marker='x', s=100, label='Depot')

    plt.title('Best Routes with traffic data')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.legend()
    plt.show()

best_fitness, best_routes, best_route_length, best_fitnesses = genetic_algorithm()

print(f"Best Route Length: {best_route_length}")

plot_routes(best_routes)
