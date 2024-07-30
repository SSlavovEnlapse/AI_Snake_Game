# utils.py

def normalize(x, max_value):
    return x / max_value if max_value else 0

def calculate_distances(snake, food, grid_size):
    head_x, head_y = snake.body[0]
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # N, S, W, E
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # NW, NE, SW, SE
    ]

    vision = []
    for d in directions:
        dist_food, dist_body, dist_wall = 0, 0, 0
        step = 1
        while True:
            x, y = head_x + d[0] * step, head_y + d[1] * step
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                dist_wall = step
                break
            if (x, y) == food:
                dist_food = step
            if (x, y) in snake.body[1:]:
                dist_body = step
                break
            step += 1
        vision.extend([normalize(dist_food, grid_size), normalize(dist_body, grid_size), normalize(dist_wall, grid_size)])
    return vision