import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gridworld_rl.env import GridWorld
from gridworld_rl.agents import QAgent
from gridworld_rl.visualizer import GridVisualizer

#Initialize Environment and Agent
env = GridWorld(height=10, width=10, stochastic=True)
state_size = env.height * env.width
action_size = 4
agent = QAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon_decay=0.9995)

# Training parameters
total_episodes = 1000

# Lists to store performance metrics
episode_rewards = []
episode_lengths = []

# Start the Training Loop
print("Starting training ...")

for episode in range(total_episodes):
    exclude_for_goal = [env.start] + list(env.obstacles)
    new_goal = env._random_free_cell(exclude=exclude_for_goal)
    env.set_goal(new_goal)
    
    exclude_for_start = list(env.obstacles) + env.goals
    new_start = env._random_free_cell(exclude=exclude_for_start)
    env.start = new_start
    env.agent.set_position(new_start)

    env.reset()
    current_pos = env.agent.pos
    state = current_pos[0] * env.width + current_pos[1]
    done = False
    total_episode_reward = 0
    steps_in_episode = 0
    while not done:
        action = agent.choose_action(state)
        obs, reward, done, actual_action = env.step(action)
        next_pos = env.agent.pos
        next_state = next_pos[0] * env.width + next_pos[1]
        agent.learn(state, actual_action, reward, next_state, done)
        state = next_state
        total_episode_reward += reward
        steps_in_episode += 1

    episode_rewards.append(total_episode_reward)
    episode_lengths.append(steps_in_episode)
    agent.decay_epsilon()

    if (episode + 1) % 100 == 0:
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}/{total_episodes} | Avg Reward (Last 100): {avg_reward_last_100:.2f} | Epsilon: {agent.epsilon:.3f}")

print("\n✅ Training finished.")

# Plot Training Performance
print("📊 Generating performance plots...")
window_size = 100
rewards_moving_avg = pd.Series(episode_rewards).rolling(window_size, min_periods=window_size).mean()

plt.figure(figsize=(12, 7))
plt.plot(episode_rewards, label='Reward per Episode', color='lightblue', alpha=0.7)
plt.plot(rewards_moving_avg, label=f'{window_size}-Episode Moving Average', color='blue', linewidth=2)
plt.title('Agent Performance: Total Reward per Episode', fontsize=16)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#Visualize Agent's Learned Behavior
print("\n🤖 Visualizing trained agent...")
visualizer = GridVisualizer(env)
visualizer.test_agent_live(agent, max_steps=100)
visualizer.plot_q_values_in_cells(agent)
