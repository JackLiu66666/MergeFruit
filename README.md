# MergeFruit - Reinforcement Learning Fruit Merge Game

## Project Introduction

This is a reinforcement learning-based "Merge Watermelon" game project, built with PyGame for the game interface and implementing DQN (Deep Q-Network) reinforcement learning algorithm. The game rules are simple: fruits of different sizes fall from the top, and two identical fruits collide to merge into a larger fruit. The ultimate goal is to synthesize a big watermelon. The project includes complete game logic, agent training code, and visualization demonstration features.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Play the Game Manually

```bash
python merge_fruit.py
```

- **A/D** or **←/→**: Move fruit
- **Space**: Drop fruit
- **R**: Restart game

### 3. Train the Agent

```bash
python train.py
```

Default training for 1000 episodes. Models and training logs will be saved in the `training_data/` directory.

### 4. Watch Agent Demonstration

```bash
python demo.py
```

- Automatically loads the latest trained model
- Right panel displays algorithm decision process and Q-values
- **↑/↓**: Adjust game speed
- **R**: Restart after game over
![HnVideoEditor_2026_03_01_234020930](https://github.com/user-attachments/assets/b52a137a-5f0b-4b11-ab2f-304c4588d50c)

  
