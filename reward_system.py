def calculate_reward(total_merged_points, fruits, game_over):

    reward = 0
    
    # 基础奖励：每次动作给予小的正向奖励
    reward += 1
    
    if total_merged_points > 0:
        # 合成奖励：根据合成的点数给予奖励
        reward += total_merged_points // 2
        
        # 高级水果奖励：根据最高等级水果给予额外奖励
        max_merged_level = 0
        for fruit in fruits:
            if fruit.type_idx > max_merged_level:
                max_merged_level = fruit.type_idx
        
        # 等级奖励，随着等级增加而递增，但幅度适中
        level_bonus = max_merged_level // 3
        reward += level_bonus
        
        # 最高等级水果奖励
        if any(f.type_idx == 10 for f in fruits):
            reward += 5
    else:
        # 没有合成时的轻微惩罚，不那么严厉
        reward -= 1
    
    # 游戏结束惩罚：适当的负面奖励
    if game_over:
        reward -= 20
    
    # 奖励标准化：确保奖励值在合理范围内
    reward = max(min(reward, 10), -25)
    
    return reward