import pygame
import sys
import os
import glob
import random
from game_logic import MergeFruitGame, WIDTH, HEIGHT, DEATH_LINE, FRUIT_TYPES, Fruit, resolve_collisions, check_merge, check_game_over
from q_agent import QLearningAgent

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BROWN = (139, 69, 19)

def load_image(name, radius):
    image_path = os.path.join("images", name + ".png")
    if os.path.exists(image_path):
        try:
            image = pygame.image.load(image_path).convert_alpha()
            image = pygame.transform.scale(image, (radius * 2, radius * 2))
            return image
        except:
            pass
    return None

def demo_with_visualization():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Merge Watermelon - Q-Learning Demo")
    clock = pygame.time.Clock()

    FRUIT_IMAGES = []
    for fruit in FRUIT_TYPES:
        FRUIT_IMAGES.append(load_image(fruit["name"], fruit["radius"]))

    BACKGROUND_IMAGE = None
    background_path = os.path.join("images", "background.png")
    if os.path.exists(background_path):
        try:
            BACKGROUND_IMAGE = pygame.image.load(background_path).convert()
            BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (WIDTH, HEIGHT))
        except:
            pass

    q_table_files = glob.glob('training_data/q_table_*.json')
    if not q_table_files:
        print("No trained Q-table found! Please run train.py first.")
        return

    latest_q_table = max(q_table_files, key=os.path.getctime)
    print(f"Loading Q-table from: {latest_q_table}")

    game = MergeFruitGame()
    action_space = game.get_action_space()
    agent = QLearningAgent(action_space)
    agent.load_q_table(latest_q_table)

    running = True
    fruits = []
    score = 0
    game_over = False
    death_timer = 0
    next_type = random.randint(0, 4)
    current_x = WIDTH // 2
    waiting_for_stable = False
    steps_since_drop = 0

    while running:
        if BACKGROUND_IMAGE is not None:
            screen.blit(BACKGROUND_IMAGE, (0, 0))
        else:
            screen.fill(GRAY)
        pygame.draw.rect(screen, BROWN, (0, HEIGHT - 20, WIDTH, 20))
        pygame.draw.line(screen, (255, 0, 0), (0, DEATH_LINE), (WIDTH, DEATH_LINE), 3)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_r:
                    fruits = []
                    score = 0
                    game_over = False
                    death_timer = 0
                    next_type = random.randint(0, 4)
                    waiting_for_stable = False
                    steps_since_drop = 0

        if not game_over:
            if not waiting_for_stable:
                state = game.get_state()
                action = agent.choose_action(state, training=False)
                col_width = WIDTH / game.get_action_space()
                x = (action + 0.5) * col_width
                x = max(FRUIT_TYPES[next_type]["radius"], min(WIDTH - FRUIT_TYPES[next_type]["radius"], x))
                
                new_fruit = Fruit(x, 120, next_type)
                new_fruit.falling = True
                fruits.append(new_fruit)
                
                next_type = random.randint(0, 4)
                waiting_for_stable = True
                steps_since_drop = 0
            else:
                steps_since_drop += 1
                all_stable = True
                for fruit in fruits:
                    if fruit.falling:
                        all_stable = False
                        break
                if all_stable or steps_since_drop > 300:
                    waiting_for_stable = False

            for fruit in fruits:
                fruit.update(fruits)
            
            resolve_collisions(fruits)
            score, _ = check_merge(fruits, score)

            game_over, death_timer = check_game_over(fruits, death_timer)

        for fruit in fruits:
            if FRUIT_IMAGES[fruit.type_idx] is not None:
                image_rect = FRUIT_IMAGES[fruit.type_idx].get_rect(center=(int(fruit.x), int(fruit.y)))
                screen.blit(FRUIT_IMAGES[fruit.type_idx], image_rect)
            else:
                pygame.draw.circle(screen, fruit.color, (int(fruit.x), int(fruit.y)), fruit.radius)
                pygame.draw.circle(screen, WHITE, (int(fruit.x) - fruit.radius // 3, int(fruit.y) - fruit.radius // 3), fruit.radius // 5)

        font = pygame.font.Font(None, 48)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        if game_over:
            game_over_font = pygame.font.Font(None, 96)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            restart_font = pygame.font.Font(None, 48)
            restart_text = restart_font.render("Press R to Restart", True, BLACK)
            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 50))
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    demo_with_visualization()
