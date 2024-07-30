import logging
import random
from snake import Snake
import numpy as np
from constants import GRID_SIZE, SNAKE_SIZE, WHITE, RED
import pygame 

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SnakeGame:
    def __init__(self, render=True):
        initial_position = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = Snake(initial_position)
        self.food = initial_position  # Place the first food at the snake's head position
        self.score = 0
        self.neural_network = None
        self.render = render  # Add render flag
        #logging.debug("Initialized SnakeGame")

    def place_food(self):
        while True:
            food_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food_pos not in self.snake.body:
                #logging.debug(f"Placed food at {food_pos}")
                return food_pos

    def update(self):
        if self.neural_network:
            vision = self.snake.look(self.food, GRID_SIZE)
            prediction = self.neural_network.forward(np.array(vision).reshape(-1, 1))
            move = np.argmax(prediction)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

            #logging.debug(f"Neural Network Prediction: {prediction.ravel()}")
            #logging.debug(f"Chosen Move: {directions[move]} based on max index {move}")

            self.snake.set_direction(directions[move])

        self.snake.move()
        if self.snake.check_collision(GRID_SIZE) or self.snake.moves <= 0:
            #logging.debug("Game over")
            return False  # Game over
        
        if self.snake.body[0] == self.food:
            self.snake.grow()
            self.food = self.place_food()
            self.score += 1
            #logging.debug(f"Ate food, new score: {self.score}")

        return True  # Continue the game

    def draw(self, screen):
        if self.render:
            # Draw game grid
            grid_width = SNAKE_SIZE * GRID_SIZE
            grid_height = SNAKE_SIZE * GRID_SIZE
            game_x_offset = 300  # Offset for the game grid
            pygame.draw.rect(screen, WHITE, [game_x_offset, 0, grid_width, grid_height], 2)

            # Draw snake
            for segment in self.snake.body:
                pygame.draw.rect(screen, WHITE, 
                                (game_x_offset + segment[1] * SNAKE_SIZE, segment[0] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
            
            # Draw food
            pygame.draw.rect(screen, RED, 
                            (game_x_offset + self.food[1] * SNAKE_SIZE, self.food[0] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))