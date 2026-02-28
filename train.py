import csv
import os
from datetime import datetime
from game_logic import MergeFruitGame
from q_agent import QLearningAgent

def train_agent(num_episodes=5000, save_interval=500):
    os.makedirs('training_data', exist_ok=True)
    
    game = MergeFruitGame()
    action_space = game.get_action_space()
    agent = QLearningAgent(action_space)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filepath = f'training_data/training_log_{timestamp}.csv'
    q_table_filepath = f'training_data/q_table_{timestamp}.json'
    
    with open(csv_filepath, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'score', 'total_reward', 'epsilon', 'steps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for episode in range(num_episodes):
            state = game.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = game.step(action)
                
                agent.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # 计算活跃单元格数量
                active_cells = sum(1 for s in state[:-1] if s != -1)
                # 打印活跃单元格数量和状态中所有非-1的元素
                non_zero_state = [s for s in state[:-1] if s != -1]
                print(f"Episode: {episode+1}/{num_episodes}, Step: {steps}, Active cells: {active_cells}, State: {non_zero_state}, Action: {action}, Reward: {reward}, Score: {info['score']}")
            
            agent.decay_epsilon()
            
            writer.writerow({
                'episode': episode + 1,
                'score': info['score'],
                'total_reward': total_reward,
                'epsilon': agent.epsilon,
                'steps': steps
            })
            
            print(f"Episode {episode + 1}/{num_episodes} completed! Score: {info['score']}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
            
            if (episode + 1) % save_interval == 0:
                agent.save_q_table(q_table_filepath)
                print(f"Q-table saved at episode {episode + 1}")
    
    agent.save_q_table(q_table_filepath)
    print(f"Training completed! Q-table saved to {q_table_filepath}")
    print(f"Training log saved to {csv_filepath}")

if __name__ == "__main__":
    train_agent(num_episodes=1000, save_interval=100)
