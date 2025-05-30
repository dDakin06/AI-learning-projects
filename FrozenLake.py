import gymnasium as gym
import numpy as np
import random


env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi") # creating the environment.


q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 0.1     # Exploration rate
episodes = 1000   # Number of training episodes

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

print("Training finished!\n")


state, _ = env.reset()
print(env.render())

for step in range(20):
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(env.render())

    if done:
        print("Episode finished.")
        break
