import pygame
import numpy as np
from game import SnakeGame
from genetic_algorithm import GeneticAlgorithm
from visualization import display_interface
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, SNAKE_SIZE
import asyncio
import tkinter as tk
from tkinter import filedialog
import os
from neural_network import NeuralNetwork

# Constants
SAVE_DIR = r"C:/Users/user/Desktop/AI_Snake/models"

increase_mut_button = pygame.Rect(10, 170, 50, 30)
decrease_mut_button = pygame.Rect(70, 170, 50, 30)  

def save_model(gen_alg, save_path):
    best_index = np.argmax(gen_alg.fitness_scores)
    best_network = gen_alg.population[best_index]
    best_network.save(save_path)

def load_model(gen_alg, load_path):
    new_population = []
    for _ in range(gen_alg.population_size):
        network = NeuralNetwork()
        network.load(load_path)
        new_population.append(network)
    gen_alg.population = new_population

async def visualize_snake(screen, gen_alg, network, generation, snake_size, highscore, save_path):
    game = SnakeGame()
    game.neural_network = network

    running = True
    clock = pygame.time.Clock()

    while running and game.update():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if increase_mut_button.collidepoint(event.pos):
                    gen_alg.mutation_rate = min(gen_alg.mutation_rate + 0.05, 1.0)  # Adjust mutation rate dynamically
                elif decrease_mut_button.collidepoint(event.pos):
                    gen_alg.mutation_rate = max(gen_alg.mutation_rate - 0.05, 0.0)

        score = game.score
        highscore = max(highscore, game.score)
        
        display_interface(screen, game, network, generation + 1, score, highscore, gen_alg.mutation_rate, snake_size)

        clock.tick(15)  # Adjust this value to change the snake's speed
        await asyncio.sleep(0)  # Yield control to the event loop

    return highscore

async def run_genetic_algorithm(gen_alg, generations, screen, save_folder, game_number):
    highscore = 0

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        await gen_alg.evaluate_fitness(SnakeGame)
        gen_alg.create_new_generation()

        best_index = np.argmax(gen_alg.fitness_scores)
        best_network = gen_alg.population[best_index]
        
        # Autosave the model after each generation
        save_path = os.path.join(save_folder, f"game{game_number}_progress.json")
        save_model(gen_alg, save_path)
        
        highscore = await visualize_snake(screen, gen_alg, best_network, generation, SNAKE_SIZE, highscore, save_path)

    return game_number

def prompt_load_file():
    root = tk.Tk()
    root.withdraw()
    load_path = filedialog.askopenfilename(title="Select Save File", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    return load_path

def get_next_game_number(save_folder):
    existing_files = os.listdir(save_folder)
    game_numbers = [int(f.split('_')[0][4:]) for f in existing_files if f.startswith('game') and f.endswith('_progress.json')]
    return max(game_numbers, default=0) + 1

async def main():
    pygame.init()  # Initialize Pygame here
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake AI Evolution')

    # Ensure the save directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Determine the next game number
    game_number = get_next_game_number(SAVE_DIR)
    
    # Prompt the user to load a previous save file
    load_path = prompt_load_file()
    
    global increase_mut_button, decrease_mut_button
    
    gen_alg = GeneticAlgorithm(population_size=3000, mutation_rate=0.2, batch_size=100, elitism_rate=0.1, max_generations=5000)  # Updated parameters
    generations = 5000  # Number of generations for the demonstration

    if load_path:
        load_model(gen_alg, load_path)
    
    await run_genetic_algorithm(gen_alg, generations, screen, SAVE_DIR, game_number)

    pygame.quit()

    # Display the save folder location on exit
    print(f"Progress saved in: {SAVE_DIR}")

if __name__ == '__main__':
    asyncio.run(main())