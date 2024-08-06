import logging
import random
from snake import Snake
import numpy as np
from constants import GRID_SIZE, SNAKE_SIZE, WHITE, RED
import pygame

class SnakeGame:
    def __init__(self, render=True, food_positions=None):
        initial_position = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = Snake(initial_position)
        self.food_positions = food_positions if food_positions is not None else []
        self.food_index = 0
        self.food = self.place_food()  # Place the first food 
        self.score = 0
        self.neural_network = None
        self.render = render

    def place_food(self):
        if self.food_index < len(self.food_positions):
            food_pos = self.food_positions[self.food_index]
            self.food_index += 1
            return food_pos
        else:
            while True:
                food_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
                if food_pos not in self.snake.body:
                    self.food_positions.append(food_pos)
                    self.food_index += 1
                    return food_pos

    def update(self):
        if self.neural_network:
            vision = self.snake.look(self.food, GRID_SIZE)
            prediction = self.neural_network.forward(np.array(vision).reshape(-1, 1))
            move = np.argmax(prediction)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

            current_direction = self.snake.direction
            new_direction = directions[move]

            if (current_direction == (0, 1) and new_direction != (0, -1)) or \
               (current_direction == (0, -1) and new_direction != (0, 1)) or \
               (current_direction == (1, 0) and new_direction != (-1, 0)) or \
               (current_direction == (-1, 0) and new_direction != (1, 0)):

                self.snake.set_direction(new_direction)

        self.snake.move()
        if self.snake.check_collision(GRID_SIZE) or self.snake.moves <= 0:
            return False  # Game over
        
        if self.snake.body[0] == self.food:
            self.snake.grow()
            self.food = self.place_food()
            self.score += 1

        return True  # Continue the game

    def draw(self, screen):
        if self.render:
            grid_width = SNAKE_SIZE * GRID_SIZE
            grid_height = SNAKE_SIZE * GRID_SIZE
            game_x_offset = 300  # Offset for the game grid
            pygame.draw.rect(screen, WHITE, [game_x_offset, 0, grid_width, grid_height], 2)

            for segment in self.snake.body:
                pygame.draw.rect(screen, WHITE, 
                                (game_x_offset + segment[1] * SNAKE_SIZE, segment[0] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))
            
            pygame.draw.rect(screen, RED, 
                            (game_x_offset + self.food[1] * SNAKE_SIZE, self.food[0] * SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE))