import pygame
import random
import math
import sys
import os

pygame.init()

WIDTH = 400
HEIGHT = 600
FPS = 60
GRAVITY = 0.5
BOUNCE = 0.6
MAX_FRUITS = 11
DEATH_LINE = 150

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Merge Watermelon")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BROWN = (139, 69, 19)

FRUIT_TYPES = [
    {"name": "cherry", "radius": 30, "color": (255, 0, 0), "points": 1},
    {"name": "strawberry", "radius": 40, "color": (255, 100, 100), "points": 3},
    {"name": "grape", "radius": 50, "color": (128, 0, 128), "points": 6},
    {"name": "orange", "radius": 60, "color": (255, 165, 0), "points": 10},
    {"name": "apple", "radius": 70, "color": (255, 69, 0), "points": 15},
    {"name": "pear", "radius": 80, "color": (255, 215, 0), "points": 21},
    {"name": "peach", "radius": 90, "color": (255, 182, 193), "points": 28},
    {"name": "pineapple", "radius": 100, "color": (255, 200, 50), "points": 36},
    {"name": "melon", "radius": 110, "color": (144, 238, 144), "points": 45},
    {"name": "watermelon", "radius": 120, "color": (0, 128, 0), "points": 55},
    {"name": "big_watermelon", "radius": 140, "color": (0, 100, 0), "points": 66}
]

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

class Fruit:
    def __init__(self, x, y, type_idx):
        self.x = x
        self.y = y
        self.type_idx = type_idx
        self.radius = FRUIT_TYPES[type_idx]["radius"]
        self.color = FRUIT_TYPES[type_idx]["color"]
        self.vx = 0
        self.vy = 0
        self.falling = False
        self.merged = False

    def update(self, fruits):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * BOUNCE
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -self.vx * BOUNCE
        if self.y + self.radius > HEIGHT - 20:
            self.y = HEIGHT - 20 - self.radius
            self.vy = -self.vy * BOUNCE
            if abs(self.vy) < 1:
                self.vy = 0
                self.falling = False

        for fruit in fruits:
            if fruit != self and not fruit.merged and not self.merged:
                dx = self.x - fruit.x
                dy = self.y - fruit.y
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = self.radius + fruit.radius

                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    total_mass = self.radius * self.radius + fruit.radius * fruit.radius
                    ratio1 = (fruit.radius * fruit.radius) / total_mass
                    ratio2 = (self.radius * self.radius) / total_mass
                    
                    self.x += nx * overlap * ratio1
                    self.y += ny * overlap * ratio1
                    fruit.x -= nx * overlap * ratio2
                    fruit.y -= ny * overlap * ratio2

                    relative_vx = self.vx - fruit.vx
                    relative_vy = self.vy - fruit.vy
                    dot_product = relative_vx * nx + relative_vy * ny

                    if dot_product < 0:
                        impulse = -(1 + BOUNCE) * dot_product / 2
                        self.vx += impulse * nx * ratio1
                        self.vy += impulse * ny * ratio1
                        fruit.vx -= impulse * nx * ratio2
                        fruit.vy -= impulse * ny * ratio2

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - 20 - self.radius, self.y))
        
        if self.falling and abs(self.vx) < 0.5 and abs(self.vy) < 0.5:
            self.vx = 0
            self.vy = 0
            self.falling = False

    def draw(self, surface):
        if FRUIT_IMAGES[self.type_idx] is not None:
            image_rect = FRUIT_IMAGES[self.type_idx].get_rect(center=(int(self.x), int(self.y)))
            surface.blit(FRUIT_IMAGES[self.type_idx], image_rect)
        else:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surface, WHITE, (int(self.x) - self.radius // 3, int(self.y) - self.radius // 3), self.radius // 5)

def check_merge(fruits, score):
    to_remove = []
    new_fruits = []

    for i, fruit1 in enumerate(fruits):
        for j, fruit2 in enumerate(fruits):
            if i < j and not fruit1.merged and not fruit2.merged and fruit1.type_idx == fruit2.type_idx:
                dx = fruit1.x - fruit2.x
                dy = fruit1.y - fruit2.y
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = fruit1.radius + fruit2.radius

                if dist < min_dist + 5:
                    fruit1.merged = True
                    fruit2.merged = True
                    to_remove.append(fruit1)
                    to_remove.append(fruit2)

                    new_x = (fruit1.x + fruit2.x) / 2
                    new_y = (fruit1.y + fruit2.y) / 2
                    new_type = fruit1.type_idx + 1

                    if new_type < MAX_FRUITS:
                        new_fruit = Fruit(new_x, new_y, new_type)
                        new_fruit.falling = True
                        new_fruits.append(new_fruit)
                        score += FRUIT_TYPES[new_type]["points"]

    for fruit in to_remove:
        if fruit in fruits:
            fruits.remove(fruit)

    fruits.extend(new_fruits)
    return score

def main():
    running = True
    fruits = []
    score = 0
    game_over = False
    death_timer = 0

    next_type = random.randint(0, 4)
    current_x = WIDTH // 2

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
            if event.type == pygame.KEYDOWN and not game_over:
                if event.key == pygame.K_SPACE:
                    new_fruit = Fruit(current_x, 120, next_type)
                    new_fruit.falling = True
                    fruits.append(new_fruit)
                    next_type = random.randint(0, 4)
                if event.key == pygame.K_r:
                    fruits = []
                    score = 0
                    game_over = False
                    death_timer = 0
                    next_type = random.randint(0, 4)
            if event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_r:
                    fruits = []
                    score = 0
                    game_over = False
                    death_timer = 0
                    next_type = random.randint(0, 4)

        if not game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                current_x -= 5
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                current_x += 5

            current_x = max(FRUIT_TYPES[next_type]["radius"], min(WIDTH - FRUIT_TYPES[next_type]["radius"], current_x))

            for fruit in fruits:
                fruit.update(fruits)

            score = check_merge(fruits, score)

            any_fruit_over_death_line = False
            for fruit in fruits:
                if fruit.y - fruit.radius < DEATH_LINE:
                    any_fruit_over_death_line = True
                    break
            
            if any_fruit_over_death_line:
                death_timer += 1
                if death_timer > 60:
                    game_over = True
            else:
                death_timer = 0

        for fruit in fruits:
            fruit.draw(screen)

        if not game_over:
            preview_fruit = Fruit(current_x, 80, next_type)
            preview_fruit.draw(screen)

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
        else:
            controls_font = pygame.font.Font(None, 32)
            controls_text = controls_font.render("A/D: Move | Space: Drop", True, BLACK)
            screen.blit(controls_text, (10, 60))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
