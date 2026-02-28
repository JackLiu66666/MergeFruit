def calculate_reward(total_merged_points, fruits, game_over):

    reward = 0
    
    reward += 2
    
    if total_merged_points > 0:
        max_merged_level = 0
        for fruit in fruits:
            if fruit.type_idx > max_merged_level:
                max_merged_level = fruit.type_idx
        
        level_bonus = max_merged_level * 5
        reward += total_merged_points + level_bonus
        
        if len([f for f in fruits if f.falling]) > 0:
            reward += 10
        
        if any(f.type_idx == 10 for f in fruits):
            reward += 50
    else:
        reward -= 10
    
    if game_over:
        reward -= 100
    
    return reward