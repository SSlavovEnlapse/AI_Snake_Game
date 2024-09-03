import pygame
from constants import GRID_SIZE, SNAKE_SIZE, WHITE, BLACK, GREEN, RED, BLUE, GRAY

def display_game(screen, game, snake_size):
    # Draw game grid
    grid_width = snake_size * GRID_SIZE
    grid_height = snake_size * GRID_SIZE
    game_x_offset = 300  # Offset for the game grid
    pygame.draw.rect(screen, WHITE, [game_x_offset, 0, grid_width, grid_height], 2)

    # Draw snake
    for segment in game.snake.body:
        pygame.draw.rect(screen, WHITE,
                         (game_x_offset + segment[1] * snake_size, segment[0] * snake_size, snake_size, snake_size))

    # Draw food
    pygame.draw.rect(screen, RED,
                     (game_x_offset + game.food[1] * snake_size, game.food[0] * snake_size, snake_size, snake_size))

def display_stats(screen, generation, score, highscore, mutation_rate):
    font = pygame.font.Font(None, 36)
    gen_text = font.render(f"Generation: {generation}", True, WHITE)
    score_text = font.render(f"Score: {score}", True, WHITE)
    highscore_text = font.render(f"High Score: {highscore}", True, WHITE)
    mut_text = font.render(f"Mutation Rate: {mutation_rate:.1%}", True, WHITE)

    screen.blit(gen_text, (10, 10))
    screen.blit(score_text, (10, 50))
    screen.blit(highscore_text, (10, 90))
    screen.blit(mut_text, (10, 130))

    # Draw the buttons for mutation rate adjustment
    pygame.draw.rect(screen, WHITE, (10, 170, 50, 30))
    pygame.draw.rect(screen, WHITE, (70, 170, 50, 30))

    incre_button_text = font.render("+", True, BLACK)
    decre_button_text = font.render("-", True, BLACK)

    screen.blit(incre_button_text, (25, 175))
    screen.blit(decre_button_text, (85, 175))

def draw_neural_network(screen, network, start_x, start_y, width, height):
    layers = network.layer_sizes
    layer_width = width // (len(layers) - 1)

    node_radius = 10  # Radius of each node
    max_nodes = max(layers)

    for l, layer_size in enumerate(layers):
        layer_x = start_x + l * layer_width
        spacing_y = height // (layer_size + 1)

        for n in range(layer_size):
            node_y = start_y + (n + 1) * spacing_y
            pygame.draw.circle(screen, WHITE, (layer_x, node_y), node_radius)

            if l < len(layers) - 1:
                next_layer_size = layers[l + 1]
                next_layer_x = start_x + (l + 1) * layer_width
                next_spacing_y = height // (next_layer_size + 1)

                for nn in range(next_layer_size):
                    next_node_y = start_y + (nn + 1) * next_spacing_y
                    weight = network.weights[l][nn, n]
                    color = RED if weight < 0 else BLUE
                    pygame.draw.line(screen, color, (layer_x, node_y), (next_layer_x, next_node_y))

def display_interface(screen, game, network, generation, score, highscore, mutation_rate, snake_size):
    screen.fill(BLACK)
    
    # Define panel dimensions
    panel_width = 300
    panel_height = screen.get_height()
    panel_x = 0
    panel_y = 0

    # Draw panel background
    pygame.draw.rect(screen, GRAY, [panel_x, panel_y, panel_width, panel_height])

    # Display neural network and stats in the panel
    draw_neural_network(screen, network, 10, 200, panel_width - 20, panel_height - 240)
    display_stats(screen, generation, score, highscore, mutation_rate)

    # Display game grid on the right
    display_game(screen, game, snake_size)
    
    pygame.display.update()