import logging
import numpy as np
import random
from copy import deepcopy
from neural_network import NeuralNetwork
import asyncio

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
            #logging.debug(f"Fitness for this game: {fitness}")

    avg_score = total_score / games_per_network if games_per_network else 1
    #logging.debug(f"Average fitness: {avg_score}")
    return avg_score

class GeneticAlgorithm:
    def __init__(self, population_size=500, mutation_rate=0.15, batch_size=50, elitism_rate=0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate  # Set a fixed mutation rate to 15%
        self.batch_size = batch_size
        self.elitism_rate = elitism_rate
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.fitness_scores = []
        #logging.debug("Initialized GeneticAlgorithm")

    async def evaluate_fitness(self, game_class, games_per_network=1):
        self.fitness_scores = []
        
        # Divide population into batches
        batches = [self.population[i:i+self.batch_size] for i in range(0, self.population_size, self.batch_size)]
        
        for batch in batches:
            tasks = [asyncio.create_task(evaluate_network(network, game_class, games_per_network, render=False)) for network in batch]
            results = await asyncio.gather(*tasks)
            self.fitness_scores.extend(results)
        
        # Normalize fitness scores
        self.fitness_scores = np.array(self.fitness_scores)
        self.fitness_scores = self.fitness_scores / np.sum(self.fitness_scores)
        #logging.debug(f"Fitness scores: {self.fitness_scores}")

    def select_parents(self):
        # Using fitness-proportionate selection (roulette wheel)
        cumulative_fitness = np.cumsum(self.fitness_scores)
        random_select = random.random() * cumulative_fitness[-1]
        parent_idx = np.searchsorted(cumulative_fitness, random_select)
        selected_parent = self.population[parent_idx]
        #logging.debug(f"Selected parent at index: {parent_idx}")
        return selected_parent

    def crossover(self, parent1, parent2):
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        for i in range(len(parent1.weights)):
           crossover_point = random.randint(0, parent1.weights[i].shape[1] - 1)
           child1.weights[i][:, :crossover_point], child2.weights[i][:, :crossover_point] = \
                deepcopy(parent2.weights[i][:, :crossover_point]), deepcopy(parent1.weights[i][:, :crossover_point])
       #logging.debug("Performed crossover")
        return child1, child2

    def mutate(self, network):
        for i in range(len(network.weights)):
            if random.random() < self.mutation_rate:
                noise = np.random.randn(*network.weights[i].shape) * 0.1
                network.weights[i] += noise
        #logging.debug("Mutated network")
        return network

    def create_new_generation(self):
        new_population = []

        # Elitism: Retain top-performing individuals
        num_elites = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.fitness_scores)[-num_elites:]
        elites = [self.population[i] for i in elite_indices]
        new_population.extend(elites)

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(), self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))

        self.population = new_population[:self.population_size]
        #logging.debug("Created new generation")