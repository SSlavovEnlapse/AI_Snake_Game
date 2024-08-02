import logging
import random
from constants import GRID_SIZE
# Configure logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Snake:
    def __init__(self, initial_position):
        self.body = [initial_position]
        self.direction = (0, 1)
        self.grow_flag = False
        self.lifetime = 0
        self.score = 0
        self.moves = 200
        self.max_moves = 500
        self.fitness = 0
        self.previous_positions = []
        self.loop_penalty = 0
        self.visited_positions = set()
        self.last_food_position = initial_position
        #logging.debug(f"Initialized snake at position {initial_position}")

    def set_direction(self, new_direction):
        self.direction = new_direction
        #logging.debug(f"Set new direction to {new_direction}")

    def move(self):
        self.lifetime += 1
        self.moves -= 1
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.body.insert(0, new_head)

        if not self.grow_flag:
            self.body.pop()
        else:
            self.grow_flag = False

        if len(self.previous_positions) > 50:
            self.previous_positions.pop(0)
        self.previous_positions.append(new_head)

        if self.previous_positions.count(new_head) > 1:
            self.loop_penalty += 2

        self.visited_positions.add(new_head)

        #logging.debug(f"Moved to {new_head}, lifetime: {self.lifetime}, moves left: {self.moves}")

    def grow(self):
        self.grow_flag = True
        self.previous_positions = []
        self.loop_penalty = 0
        self.moves = min(self.moves + 100, self.max_moves)
        #logging.debug(f"Grew, new moves count: {self.moves}")

    def check_collision(self, grid_size):
        head_x, head_y = self.body[0]
        if not (0 <= head_x < grid_size and 0 <= head_y < grid_size):
            return True
        if (head_x, head_y) in self.body[1:]:
            return True
        return False

    def look(self, food, grid_size):
        vision = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for direction in directions:
            vision.extend(self.look_in_direction(direction, food, grid_size))
        return vision

    def look_in_direction(self, direction, food, grid_size):
        look = [0, 0, 0]
        pos = [self.body[0][0], self.body[0][1]]
        distance = 0
        food_found = False
        body_found = False
        
        while 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
            pos[0] += direction[0]
            pos[1] += direction[1]
            distance += 1

            if not food_found and (pos[0], pos[1]) == food:
                food_found = True
                look[0] = 1 / distance
            if not body_found and (pos[0], pos[1]) in self.body[1:]:
                body_found = True
                look[1] = 1 / distance

        look[2] = 1 / distance if distance > 0 else 0
        return look

    def calculate_fitness(self, food_position):
        #logging.debug(f"Calculating fitness for snake with score: {self.score} and lifetime: {self.lifetime}")

        # Base fitness influenced by score and lifetime
        if self.score < 10:
            self.fitness = self.lifetime * (2 ** self.score)
        else:
            self.fitness = self.lifetime
            self.fitness *= 2 ** 10
            self.fitness *= (self.score - 9)

        #logging.debug(f"Base fitness (without penalties and bonuses): {self.fitness}")

        # Apply a more balanced loop penalty
        self.fitness -= self.loop_penalty * 50
        #logging.debug(f"Fitness after loop penalty ({self.loop_penalty}): {self.fitness}")

        # Distance to food bonus
        head_x, head_y = self.body[0]
        food_x, food_y = food_position
        distance_to_food = abs(head_x - food_x) + abs(head_y - food_y)
        #logging.debug(f"Distance to food: {distance_to_food}")

        self.fitness += (1 / (distance_to_food + 1)) * 500
        #logging.debug(f"Fitness after food distance bonus: {self.fitness}")

        # Visited positions bonus
        self.fitness += len(self.visited_positions) * 50
        #logging.debug(f"Fitness after visited positions bonus ({len(self.visited_positions)}): {self.fitness}")

        # Increase the reward for eating food
        self.fitness += self.score * 5000
        #logging.debug(f"Fitness after eating food bonus: {self.fitness}")

        # Apply a smaller collision penalty
        if self.check_collision(GRID_SIZE):
            self.fitness -= 200
            #logging.debug("Applied collision penalty")

        #logging.debug(f"Final fitness score: {self.fitness}")
        return self.fitness


    def replay_move(self, food):
        self.lifetime += 1
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.body.insert(0, new_head)

        if not self.grow_flag:
            self.body.pop()
        else:
            self.grow_flag = False

        if self.body[0] == food:
            self.grow()
            if self.food_positions:
                food = self.food_positions.pop(0)
            else:
                self.replay = False
        return food
