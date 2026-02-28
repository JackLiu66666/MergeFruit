import random
import math
import os
from reward_system import calculate_reward

WIDTH = 600
HEIGHT = 900
GRAVITY = 0.5
BOUNCE = 0.2
DAMPING = 0.95
WALL_FRICTION = 0.7
MAX_FRUITS = 11
DEATH_LINE = 150
COLLISION_ITERATIONS = 5
GRID_COLS = 20
GRID_ROWS = 30

FRUIT_TYPES = [
    {"name": "grape", "radius": 30, "color": (255, 0, 0), "points": 1},
    {"name": "strawberry", "radius": 40, "color": (255, 100, 100), "points": 3},
    {"name": "lemon", "radius": 50, "color": (128, 0, 128), "points": 6},
    {"name": "orange", "radius": 60, "color": (255, 165, 0), "points": 10},
    {"name": "kiwi", "radius": 70, "color": (255, 69, 0), "points": 15},
    {"name": "tomato", "radius": 80, "color": (255, 215, 0), "points": 21},
    {"name": "peach", "radius": 90, "color": (255, 182, 193), "points": 28},
    {"name": "pineapple", "radius": 100, "color": (255, 200, 50), "points": 36},
    {"name": "coconut", "radius": 110, "color": (144, 238, 144), "points": 45},
    {"name": "half-watermelon", "radius": 120, "color": (0, 128, 0), "points": 55},
    {"name": "big_watermelon", "radius": 140, "color": (0, 100, 0), "points": 66}
]

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
        self.vx *= DAMPING
        self.vy *= DAMPING
        self.x += self.vx
        self.y += self.vy

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * BOUNCE
            self.vy *= WALL_FRICTION
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -self.vx * BOUNCE
            self.vy *= WALL_FRICTION
        if self.y + self.radius > HEIGHT - 20:
            self.y = HEIGHT - 20 - self.radius
            self.vy = -self.vy * BOUNCE
            self.vx *= WALL_FRICTION
            if abs(self.vy) < 0.3:
                self.vy = 0
                self.falling = False

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - 20 - self.radius, self.y))
        
        if self.falling and abs(self.vx) < 0.3 and abs(self.vy) < 0.3:
            self.vx = 0
            self.vy = 0
            self.falling = False

def resolve_collisions(fruits):
    # 只处理未合并的水果
    active_fruits = [fruit for fruit in fruits if not fruit.merged]
    num_fruits = len(active_fruits)
    
    for _ in range(COLLISION_ITERATIONS):
        for i in range(num_fruits):
            fruit1 = active_fruits[i]
            for j in range(i + 1, num_fruits):
                fruit2 = active_fruits[j]
                dx = fruit1.x - fruit2.x
                dy = fruit1.y - fruit2.y
                dist_sq = dx * dx + dy * dy
                min_dist = fruit1.radius + fruit2.radius
                min_dist_sq = min_dist * min_dist

                if dist_sq < min_dist_sq and dist_sq > 0:
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    total_mass = fruit1.radius * fruit1.radius + fruit2.radius * fruit2.radius
                    ratio1 = (fruit2.radius * fruit2.radius) / total_mass
                    ratio2 = (fruit1.radius * fruit1.radius) / total_mass
                    
                    fruit1.x += nx * overlap * ratio1
                    fruit1.y += ny * overlap * ratio1
                    fruit2.x -= nx * overlap * ratio2
                    fruit2.y -= ny * overlap * ratio2

                    relative_vx = fruit1.vx - fruit2.vx
                    relative_vy = fruit1.vy - fruit2.vy
                    dot_product = relative_vx * nx + relative_vy * ny

                    if dot_product < 0:
                        impulse = -(1 + BOUNCE) * dot_product / 2
                        fruit1.vx += impulse * nx * ratio1
                        fruit1.vy += impulse * ny * ratio1
                        fruit2.vx -= impulse * nx * ratio2
                        fruit2.vy -= impulse * ny * ratio2

        for fruit in active_fruits:
            fruit.x = max(fruit.radius, min(WIDTH - fruit.radius, fruit.x))
            fruit.y = max(fruit.radius, min(HEIGHT - 20 - fruit.radius, fruit.y))

def check_merge(fruits, score):
    to_remove = []
    new_fruits = []
    merged_points = 0
    merge_count = 0
    max_level_merge_count = 0

    # 只处理未合并的水果
    active_fruits = [fruit for fruit in fruits if not fruit.merged]
    num_fruits = len(active_fruits)

    for i in range(num_fruits):
        fruit1 = active_fruits[i]
        if fruit1.merged:
            continue
        for j in range(i + 1, num_fruits):
            fruit2 = active_fruits[j]
            if fruit2.merged or fruit1.type_idx != fruit2.type_idx:
                continue
            
            dx = fruit1.x - fruit2.x
            dy = fruit1.y - fruit2.y
            dist_sq = dx * dx + dy * dy
            min_dist = fruit1.radius + fruit2.radius
            min_dist_sq = (min_dist + 5) ** 2

            if dist_sq < min_dist_sq:
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
                    points = FRUIT_TYPES[new_type]["points"]
                    score += points
                    merged_points += points
                    merge_count += 1
                    if new_type == MAX_FRUITS - 1:
                        max_level_merge_count += 1

    for fruit in to_remove:
        if fruit in fruits:
            fruits.remove(fruit)

    fruits.extend(new_fruits)
    return score, merged_points, merge_count, max_level_merge_count

def get_grid_state(fruits):
    grid = [[-1 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    cell_width = WIDTH / GRID_COLS
    cell_height = (HEIGHT - 20) / GRID_ROWS

    for fruit in fruits:
        # 计算水果覆盖的所有单元格
        fruit_radius = fruit.radius
        min_x = max(0, fruit.x - fruit_radius)
        max_x = min(WIDTH, fruit.x + fruit_radius)
        min_y = max(0, fruit.y - fruit_radius)
        max_y = min(HEIGHT - 20, fruit.y + fruit_radius)

        # 计算覆盖的列范围
        min_col = int(min_x / cell_width)
        max_col = int(max_x / cell_width)
        # 计算覆盖的行范围
        min_row = int(min_y / cell_height)
        max_row = int(max_y / cell_height)

        # 确保索引在有效范围内
        min_col = max(0, min_col)
        max_col = min(GRID_COLS - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(GRID_ROWS - 1, max_row)

        # 标记所有被覆盖的单元格，保留较大的水果
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # 只有当当前单元格为空或新水果更大时才更新
                if grid[row][col] < fruit.type_idx:
                    grid[row][col] = fruit.type_idx

    flat_grid = []
    for row in grid:
        flat_grid.extend(row)
    return flat_grid

def check_game_over(fruits, death_timer):
    any_fruit_over_death_line = False
    for fruit in fruits:
        if fruit.y - fruit.radius < DEATH_LINE:
            any_fruit_over_death_line = True
            break
    
    if any_fruit_over_death_line:
        death_timer += 1
        game_over = death_timer > 60
    else:
        death_timer = 0
        game_over = False
    
    return game_over, death_timer

def check_stable(fruits):
    for fruit in fruits:
        if fruit.falling:
            return False
    return True

class MergeFruitGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fruits = []
        self.score = 0
        self.game_over = False
        self.death_timer = 0
        self.next_type = random.randint(0, 4)
        self.merge_count = 0  # 成功合成的水果次数
        self.max_level_merge_count = 0  # 最高等级水果合成次数
        return self.get_state()

    def get_state(self):
        grid_state = get_grid_state(self.fruits)
        # 计算活跃单元格数量
        active_cells = sum(1 for s in grid_state if s != -1)
        # 打印水果数量和活跃单元格数量
        # print(f"Fruits: {len(self.fruits)}, Active cells: {active_cells}")
        state = grid_state + [self.next_type]
        return tuple(state)

    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True, {}

        col_width = WIDTH / GRID_COLS
        x = (action + 0.5) * col_width
        x = max(FRUIT_TYPES[self.next_type]["radius"], min(WIDTH - FRUIT_TYPES[self.next_type]["radius"], x))

        new_fruit = Fruit(x, 120, self.next_type)
        new_fruit.falling = True
        self.fruits.append(new_fruit)
        
        self.next_type = random.randint(0, 4)
        
        total_merged_points = 0
        stable = False
        max_iterations = 100
        iteration = 0
        
        while not stable and iteration < max_iterations:
            # 只更新未合并的水果
            for fruit in self.fruits:
                if not fruit.merged:
                    fruit.update(self.fruits)
            resolve_collisions(self.fruits)
            self.score, merged_points, merge_count, max_level_merge_count = check_merge(self.fruits, self.score)
            total_merged_points += merged_points
            self.merge_count += merge_count
            self.max_level_merge_count += max_level_merge_count
            
            if check_stable(self.fruits):
                stable = True
            iteration += 1

        self.game_over, self.death_timer = check_game_over(self.fruits, self.death_timer)

        # 使用奖励系统计算奖励
        reward = calculate_reward(total_merged_points, self.fruits, self.game_over)
        
        return self.get_state(), reward, self.game_over, {"score": self.score, "merge_count": self.merge_count, "max_level_merge_count": self.max_level_merge_count}

    def get_action_space(self):
        return GRID_COLS
