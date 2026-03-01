import pygame
import sys
import os
import glob
import random
import torch
from game_logic import MergeFruitGame, WIDTH, HEIGHT, DEATH_LINE, FRUIT_TYPES, Fruit, resolve_collisions, check_merge, check_game_over, get_grid_state, GRID_COLS, GRID_ROWS
from dqn_agent import DQNAgent

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BROWN = (139, 69, 19)

PREVIEW_WIDTH = 300
TOTAL_WIDTH = WIDTH + PREVIEW_WIDTH

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

def draw_grid_overlay(screen, fruits):
    cell_width = WIDTH / GRID_COLS
    cell_height = (HEIGHT - 20) / GRID_ROWS
    
    grid_state = get_grid_state(fruits)
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            type_idx = grid_state[row * GRID_COLS + col]
            if type_idx != -1:
                fruit_color = FRUIT_TYPES[type_idx]["color"]
                r, g, b = fruit_color
                
                x = col * cell_width
                y = row * cell_height
                w = cell_width
                h = cell_height
                
                overlay_surface = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
                overlay_surface.fill((r, g, b, 40))
                screen.blit(overlay_surface, (int(x), int(y)))
                
                pygame.draw.rect(screen, (50, 50, 50, 50), (int(x), int(y), int(w), int(h)), 1)

def draw_algorithm_preview(screen, q_values, chosen_action, next_type, game_speed):
    preview_x = WIDTH
    preview_width = PREVIEW_WIDTH
    preview_height = HEIGHT
    
    # 绘制预览面板背景
    pygame.draw.rect(screen, (230, 230, 230), (preview_x, 0, preview_width, preview_height))
    pygame.draw.line(screen, (100, 100, 100), (preview_x, 0), (preview_x, preview_height), 2)
    
    # 绘制标题
    font = pygame.font.Font(None, 36)
    title_text = font.render("Algorithm Preview", True, BLACK)
    screen.blit(title_text, (preview_x + 20, 20))
    
    # 绘制下一个水果
    next_fruit_text = font.render(f"Next Fruit: {FRUIT_TYPES[next_type]['name']}", True, BLACK)
    screen.blit(next_fruit_text, (preview_x + 20, 60))
    
    # 绘制当前速度
    speed_text = font.render(f"Speed: {game_speed}x", True, BLACK)
    screen.blit(speed_text, (preview_x + 20, 90))
    
    # 绘制 Q 值
    if q_values is not None:
        q_values_text = font.render("Q Values:", True, BLACK)
        screen.blit(q_values_text, (preview_x + 20, 130))
        
        cell_height = 25
        max_q = max(q_values) if len(q_values) > 0 else 1
        min_q = min(q_values) if len(q_values) > 0 else 0
        q_range = max_q - min_q if max_q != min_q else 1
        
        for i in range(len(q_values)):
            q_value = q_values[i]
            # 计算颜色强度（基于 Q 值）
            intensity = (q_value - min_q) / q_range
            green = int(255 * intensity)
            red = int(255 * (1 - intensity))
            color = (red, green, 0)
            
            # 绘制 Q 值条
            bar_width = preview_width - 40
            bar_height = 20
            bar_x = preview_x + 20
            bar_y = 170 + i * cell_height
            
            # 绘制背景条
            pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
            # 绘制 Q 值条
            bar_fill_width = bar_width * intensity
            pygame.draw.rect(screen, color, (bar_x, bar_y, bar_fill_width, bar_height))
            
            # 绘制格子编号和 Q 值
            text = font.render(f"{i}: {q_value:.2f}", True, BLACK)
            screen.blit(text, (bar_x + 5, bar_y + 2))
            
            # 标记选中的格子
            if i == chosen_action:
                pygame.draw.rect(screen, (0, 0, 255), (bar_x - 5, bar_y - 5, bar_width + 10, bar_height + 10), 2)
    
    # 绘制说明
    explanation_font = pygame.font.Font(None, 24)
    explanation_texts = [
        "Model Decision Process:",
        "1. Computes Q-values for all 20 positions",
        "2. Selects position with highest Q-value",
        "3. Q-value indicates expected future reward",
        "4. Green = higher Q-value",
        "5. Red = lower Q-value",
        "",
        "Speed Control:",
        "↑ = Increase speed",
        "↓ = Decrease speed"
    ]
    
    y_offset = 170 + len(q_values) * cell_height + 40 if q_values is not None else 170
    for i, text in enumerate(explanation_texts):
        explanation_surface = explanation_font.render(text, True, BLACK)
        screen.blit(explanation_surface, (preview_x + 20, y_offset + i * 30))

def demo_with_visualization():
    pygame.init()
    screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
    pygame.display.set_caption("Merge Watermelon - Q-Learning Demo with Algorithm Preview")
    clock = pygame.time.Clock()
    
    # 添加速度控制
    game_speed = 1.0  # 默认速度
    max_speed = 5.0   # 最大速度
    min_speed = 1   # 最小速度
    speed_step = 1  # 速度调整步长

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

    model_files = glob.glob('training_data/dqn_model_*.pt')
    if not model_files:
        print("No trained DQN model found! Please run train.py first.")
        return

    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading DQN model from: {latest_model}")

    game = MergeFruitGame()
    action_space = game.get_action_space()
    state = game.reset()
    state_size = len(state)
    
    agent = DQNAgent(state_size, action_space)
    agent.load_model(latest_model)
    print("Model loaded successfully!")

    running = True
    fruits = []
    score = 0
    game_over = False
    death_timer = 0
    next_type = random.randint(0, 4)
    current_x = WIDTH // 2
    waiting_for_stable = False
    steps_since_drop = 0
    q_values = None
    chosen_action = None

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
            # 添加速度调整按键
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game_speed = min(game_speed + speed_step, max_speed)
                elif event.key == pygame.K_DOWN:
                    game_speed = max(game_speed - speed_step, min_speed)

        if not game_over:
            if not waiting_for_stable:
                # 更新游戏状态以匹配当前水果
                game.fruits = fruits
                game.score = score
                game.death_timer = death_timer
                game.next_type = next_type
                
                state = game.get_state()
                q_values = agent.get_q_values(state)
                action = agent.choose_action(state, training=False)
                chosen_action = action
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
            score, _, _, _ = check_merge(fruits, score)

            game_over, death_timer = check_game_over(fruits, death_timer)

        for fruit in fruits:
            if FRUIT_IMAGES[fruit.type_idx] is not None:
                image_rect = FRUIT_IMAGES[fruit.type_idx].get_rect(center=(int(fruit.x), int(fruit.y)))
                screen.blit(FRUIT_IMAGES[fruit.type_idx], image_rect)
            else:
                pygame.draw.circle(screen, fruit.color, (int(fruit.x), int(fruit.y)), fruit.radius)
                pygame.draw.circle(screen, WHITE, (int(fruit.x) - fruit.radius // 3, int(fruit.y) - fruit.radius // 3), fruit.radius // 5)
        
        draw_grid_overlay(screen, fruits)
        draw_algorithm_preview(screen, q_values, chosen_action, next_type, game_speed)

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
        clock.tick(60 * game_speed)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    demo_with_visualization()
