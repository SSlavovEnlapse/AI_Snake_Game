import pygame
import numpy as np
from game import SnakeGame
from genetic_algorithm import GeneticAlgorithm
from visualization import display_interface
from button_f import Button_b  # Import the button class
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, SNAKE_SIZE
import asyncio
import logging

increase_mut_button = pygame.Rect(340, 85, 20, 20)
decrease_mut_button = pygame.Rect(365, 85, 20, 20)

async def visualize_snake(screen, gen_alg, network, generation, snake_size, highscore, human_control=False):
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
                    gen_alg.mutation_rate *= 2
                elif decrease_mut_button.collidepoint(event.pos):
                    gen_alg.mutation_rate /= 2
                elif save_button.is_clicked(event):
                    pass
                elif load_button.is_clicked(event):
                    pass
                elif human_control_button.is_clicked(event):
                    human_control = not human_control  # Toggle human control mode

        if human_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                game.snake.set_direction((-1, 0))
            if keys[pygame.K_DOWN]:
                game.snake.set_direction((1, 0))
            if keys[pygame.K_LEFT]:
                game.snake.set_direction((0, -1))
            if keys[pygame.K_RIGHT]:
                game.snake.set_direction((0, 1))

        score = game.score
        highscore = max(highscore, game.score)
        
        head_pos = game.snake.body[0]
        #logging.debug(f"Best snake head position: {head_pos}, direction: {game.snake.direction}")

        display_interface(screen, game, network, generation + 1, score, highscore, gen_alg.mutation_rate, snake_size)
        clock.tick(15)  # Adjust this value to change the snake's speed
        await asyncio.sleep(0)  # Yield control to the event loop

    return highscore

async def run_genetic_algorithm(gen_alg, generations, screen):
    highscore = 0

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        await gen_alg.evaluate_fitness(SnakeGame)
        gen_alg.create_new_generation()

        best_index = np.argmax(gen_alg.fitness_scores)
        best_network = gen_alg.population[best_index]
        
        #logging.debug(f"Generation {generation + 1}: Best fitness score: {gen_alg.fitness_scores[best_index]} at index {best_index}")

        highscore = await visualize_snake(screen, gen_alg, best_network, generation, SNAKE_SIZE, highscore, human_control=False)

async def main():
    pygame.init()  # Initialize Pygame here
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Snake AI Evolution')

    global save_button, load_button, human_control_button
    save_button = Button_b(149, 15, 100, 30, "Save")
    load_button = Button_b(249, 15, 30, 100, "Load")
    human_control_button = Button_b(449, 15, 150, 30, "Human Control")
    
    gen_alg = GeneticAlgorithm(population_size=2000)  # Increase the population size
    generations = 500  # Number of generations for the demonstration

    await run_genetic_algorithm(gen_alg, generations, screen)

    pygame.quit()

if __name__ == '__main__':
    asyncio.run(main())