import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.norm = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Dueling DQN分支
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_space, learning_rate=0.001, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=20000):
        self.state_size = state_size
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建主网络和目标网络
        self.policy_net = DQN(state_size, action_space).to(self.device)
        self.target_net = DQN(state_size, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def get_q_values(self, state):
        """
        获取给定状态下所有动作的 Q 值
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.squeeze().cpu().numpy()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, num_updates=4):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        total_loss = 0.0
        for _ in range(num_updates):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # 计算当前状态的Q值
            current_q = self.policy_net(states).gather(1, actions).squeeze(1)
            
            # 计算下一个状态的最大Q值
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
            
            # 计算损失并更新网络
            loss = self.criterion(current_q, target_q)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss / num_updates
    
    def update_target_network(self, tau=0.001):
        """
        使用软更新策略更新目标网络
        tau: 软更新参数，控制目标网络更新的速度
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath):
        # 加载模型权重，使用strict=False来处理模型架构变化
        state_dict = torch.load(filepath)
        
        # 检查是否缺少LayerNorm参数
        if "norm.weight" not in state_dict:
            # 对于旧模型，只加载存在的参数
            self.policy_net.load_state_dict(state_dict, strict=False)
        else:
            # 对于新模型，正常加载
            self.policy_net.load_state_dict(state_dict)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())