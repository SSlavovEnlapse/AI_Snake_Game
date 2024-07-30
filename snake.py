import logging

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Snake:
    def __init__(self, initial_position):
        self.body = [initial_position]
        self.direction = (0, 1)
        self.grow_flag = False
        self.lifetime = 0
        self.score = 3  # Start with score 3
        self.moves = 200  # Initial number of moves
        self.max_moves = 500
        self.fitness = 0  # Ensure the fitness attribute is initialized
        self.previous_positions = []
        self.loop_penalty = 0
        self.visited_positions = set()  # Track visited positions
        self.last_food_position = initial_position  # Track last food position
        #logging.debug(f"Initialized snake at position {initial_position}")

    def set_direction(self, new_direction):
        if (new_direction[0] + self.direction[0], new_direction[1] + self.direction[1]) != (0, 0):
            self.direction = new_direction
            #logging.debug(f"Set new direction to {new_direction}")

    def move(self):
        self.lifetime += 1  # Increment lifetime on each move
        self.moves -= 1  # Decrease moves on each move
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.body.insert(0, new_head)

        if not self.grow_flag:
            self.body.pop()
        else:
            self.grow_flag = False

        # Log the position
        if len(self.previous_positions) > 50:  # Keep last 50 positions
            self.previous_positions.pop(0)
        self.previous_positions.append(new_head)

        # Detect loop by checking if the new head position was already visited
        if self.previous_positions.count(new_head) > 1:
            self.loop_penalty += 2  # Increase penalty factor

        # Track visited positions to encourage exploration
        self.visited_positions.add(new_head)

       # logging.debug(f"Moved to {new_head}, lifetime: {self.lifetime}, moves left: {self.moves}")

    def grow(self):
        self.grow_flag = True
        # Reset loop-related variables
        self.previous_positions = []
        self.loop_penalty = 0
        #logging.debug(f"Ate food, resetting loop penalty and previous positions.")

        # Increase moves when eating food with a maximum limit
        self.moves = min(self.moves + 100, self.max_moves)
        #logging.debug(f"Grew, new moves count: {self.moves}")

    def check_collision(self, grid_size):
        head_x, head_y = self.body[0]
        if not (0 <= head_x < grid_size and 0 <= head_y < grid_size):
            #logging.debug("Collided with wall")
            return True
        if (head_x, head_y) in self.body[1:]:
            #logging.debug("Collided with body")
            return True
        return False

    def look(self, food, grid_size):
        vision = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # N, S, W, E
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # NW, NE, SW, SE
        ]

        for direction in directions:
            vision.extend(self.look_in_direction(direction, food, grid_size))
        #logging.debug(f"Vision: {vision}")
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

            #logging.debug(f"Checking position {pos} for food {food}")

            if not food_found and (pos[0], pos[1]) == food:
                food_found = True
                look[0] = 1 / distance
                #logging.debug(f"Food found at {pos} in direction {direction}")
            if not body_found and (pos[0], pos[1]) in self.body[1:]:
                body_found = True
                look[1] = 1 / distance
                #logging.debug(f"Body found at {pos} in direction {direction}")

        look[2] = 1 / distance if distance > 0 else 0
        #logging.debug(f"Look in direction {direction}: {look}")
        return look

    def calculate_fitness(self, food_position):
        if self.score < 10:
            self.fitness = self.lifetime ** 2 * (2 ** self.score)
        else:
            self.fitness = self.lifetime ** 2
            self.fitness *= 2 ** 10
            self.fitness *= (self.score - 9)

        # Penalize for looping
        self.fitness -= self.loop_penalty * 20  # Further increase penalty factor

        # Reward for exploration
        self.fitness += len(self.visited_positions) * 30  # Increase reward for unique visited positions

        # Additional reward for getting closer to food
        head_x, head_y = self.body[0]
        food_x, food_y = food_position
        distance_to_food = abs(head_x - food_x) + abs(head_y - food_y)
        self.fitness += (1 / (distance_to_food + 1)) * 100

        #logging.debug(f"Calculated fitness: {self.fitness}")
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