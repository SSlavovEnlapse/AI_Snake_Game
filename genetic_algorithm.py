import logging
import numpy as np
import random
from copy import deepcopy
from neural_network import NeuralNetwork
import asyncio

async def evaluate_network(network, game_class, games_per_network, render=False):
    total_score = 0
    for _ in range(games_per_network):
        game = game_class(render=render)
        game.neural_network = deepcopy(network)

        while game.update():
            await asyncio.sleep(0)  # Yield control to the event loop

        fitness = game.snake.calculate_fitness(game.food)
        if fitness is not None:
            total_score += fitness

    avg_score = total_score / games_per_network if games_per_network else 1
    return avg_score

class GeneticAlgorithm:
    def __init__(self, population_size=3000, mutation_rate=0.2, batch_size=100, elitism_rate=0.1, max_generations=5000):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.elitism_rate = elitism_rate
        self.population = [NeuralNetwork(hidden_layer_sizes=[32, 32]) for _ in range(population_size)]
        self.fitness_scores = []
        self.generation = 0
        self.max_generations = max_generations

    def generate_food_positions(self, num_positions, grid_size):
        self.food_positions.clear()
        for _ in range(num_positions):
            while True:
                food_pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
                if food_pos not in self.food_positions:
                    self.food_positions.append(food_pos)
                    break

    async def evaluate_fitness(self, game_class, games_per_network=1):
        self.fitness_scores = []
        batches = [self.population[i:i + self.batch_size] for i in range(0, self.population_size, self.batch_size)]
        for batch in batches:
            tasks = [asyncio.create_task(evaluate_network(network, game_class, games_per_network, render=False)) for network in batch]
            results = await asyncio.gather(*tasks)
            self.fitness_scores.extend(results)
        self.fitness_scores = np.array(self.fitness_scores)
        if np.sum(self.fitness_scores) > 0:
            self.fitness_scores = self.fitness_scores / np.sum(self.fitness_scores)
        else:
            logging.warning("Sum of fitness scores is zero or negative. Check fitness calculation.")

    def tournament_selection(self, tournament_size=50):  # Increased tournament size
        tournament = random.sample(list(zip(self.population, self.fitness_scores)), tournament_size)
        best_individual = max(tournament, key=lambda x: x[1])[0]
        return best_individual


    def crossover(self, parent1, parent2):
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        for i in range(len(parent1.weights)):
            crossover_point = random.randint(0, parent1.weights[i].shape[1] - 1)
            child1.weights[i][:, :crossover_point], child2.weights[i][:, :crossover_point] = \
                deepcopy(parent2.weights[i][:, :crossover_point]), deepcopy(parent1.weights[i][:, :crossover_point])
        return child1, child2

    def mutate(self, network):
        for i in range(len(network.weights)):
            if random.random() < self.mutation_rate:
                noise = np.random.randn(*network.weights[i].shape) * 0.1
                network.weights[i] += noise

        return network

    def create_new_generation(self):
        new_population = []
        num_elites = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.fitness_scores)[-num_elites:]
        elites = [self.population[i] for i in elite_indices]
        new_population.extend(elites)
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))
        self.population = new_population[:self.population_size]
        self.generation += 1