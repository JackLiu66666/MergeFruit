import json
import random
import os

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.9995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q_table = {}

    def get_q_value(self, state, action):
        return self.Q_table.get((state, action), 0.0)

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        
        q_values = [self.get_q_value(state, action) for action in range(self.action_space)]
        max_q = max(q_values)
        best_actions = [action for action, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state, done):
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, next_action) for next_action in range(self.action_space)]
            target_q = reward + self.discount_factor * max(next_q_values)
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.Q_table[(state, action)] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_table = {}
        for (state, action), value in self.Q_table.items():
            key = f"{state}_{action}"
            serializable_table[key] = value
        with open(filepath, 'w') as f:
            json.dump(serializable_table, f)

    def load_q_table(self, filepath):
        with open(filepath, 'r') as f:
            serializable_table = json.load(f)
        self.Q_table = {}
        for key, value in serializable_table.items():
            state_str, action_str = key.rsplit('_', 1)
            state = tuple(map(int, state_str.strip('()').split(',')))
            action = int(action_str)
            self.Q_table[(state, action)] = value
