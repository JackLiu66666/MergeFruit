import csv
import os
from datetime import datetime
from game_logic import MergeFruitGame
from dqn_agent import DQNAgent

def train_agent(num_episodes=5000, save_interval=500, target_update=10):
    os.makedirs('training_data', exist_ok=True)
    
    game = MergeFruitGame()
    action_space = game.get_action_space()
    state = game.reset()
    state_size = len(state)
    
    agent = DQNAgent(state_size, action_space)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filepath = f'training_data/training_log_{timestamp}.csv'
    model_filepath = f'training_data/dqn_model_{timestamp}.pt'
    
    with open(csv_filepath, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'score', 'total_reward', 'epsilon', 'steps', 'avg_loss', 'merge_rate', 'max_level_merges']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for episode in range(num_episodes):
            state = game.reset()
            total_reward = 0
            steps = 0
            done = False
            total_loss = 0.0
            loss_count = 0
            
            while not done:
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = game.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.learn(num_updates=4)
                if loss > 0:
                    total_loss += loss
                    loss_count += 1
                
                # 每次step都进行目标网络软更新
                agent.update_target_network()
                
                state = next_state
                total_reward += int(reward)
                steps += 1
                
                # # 计算活跃单元格数量
                # active_cells = sum(1 for s in state[:-1] if s != -1)
                # if steps % 15 == 0:
                #     non_zero_state = [s for s in state[:-1] if s != -1]
                #     print(f"Episode: {episode+1}/{num_episodes}, Step: {steps}, Active cells: {active_cells}, State: {non_zero_state}, Action: {action}, Reward: {reward}, Score: {info['score']}")
            
            agent.decay_epsilon()
            
            # 计算平均损失
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            
            # 计算合成率
            merge_count = info.get('merge_count', 0)
            merge_rate = (merge_count / steps * 100) if steps > 0 else 0.0
            
            # 获取最高等级水果合成次数
            max_level_merges = info.get('max_level_merge_count', 0)
            
            writer.writerow({
                'episode': episode + 1,
                'score': info['score'],
                'total_reward': total_reward,
                'epsilon': agent.epsilon,
                'steps': steps,
                'avg_loss': avg_loss,
                'merge_rate': merge_rate,
                'max_level_merges': max_level_merges
            })
            
            print(f"Episode {episode + 1}/{num_episodes} completed! Score: {info['score']}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Avg Loss: {avg_loss:.4f}, Merge Rate: {merge_rate:.2f}%, Max Level Merges: {max_level_merges}")
            
            if (episode + 1) % save_interval == 0:
                agent.save_model(model_filepath)
                print(f"Model saved at episode {episode + 1}")
    
    agent.save_model(model_filepath)
    print(f"Training completed! Model saved to {model_filepath}")
    print(f"Training log saved to {csv_filepath}")

if __name__ == "__main__":
    train_agent(num_episodes=1000, save_interval=100, target_update=10)
