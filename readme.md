<div align="center">

# 🤖 Q-Learning Agent in a Dynamic GridWorld

**A complete, self-contained Python project implementing a Q-Learning agent that masters a dynamic and stochastic grid environment.**

</div>

This file contains the entire project, including all Python code and setup instructions. The agent learns an optimal policy not just for one maze, but for *any* maze by training with a **random starting position** and a **random goal** in every single episode.

---

## 🚀 Visual Demonstrations

| Agent Performance (Learning Curve) | Trained Agent in Action | "Brain" of the Agent (Q-Values) |
| :---: | :---: | :---: |
| ![Agent Learning Curve](image_3e5dc2.png) | *(This is where a GIF of your `test_agent_live` output would go)* | *(This is where a screenshot of your `plot_q_values_in_cells` output would go)* |

---

## ✨ Key Features

* **Dynamic Environment**: The agent isn't given a fixed start and goal. It must learn to solve the grid from **any** start point to **any** goal.
* **Stochastic Actions**: Set to `stochastic=True`, actions have an 80% success rate, forcing the agent to learn a robust policy that avoids risks.
* **Modular, Packaged Code**: The project is structured logically into `env`, `agents`, and `visualizer` components.
* **Advanced Visualization**:
    * **Live Agent Testing**: Watch your trained agent navigate the grid in real-time.
    * **Performance Plotting**: Automatically generates the `Reward vs. Episode` graph (shown above) to prove learning.
    * **Q-Table Inspection**: A custom plot shows all 4 Q-values (Up, Down, Left, Right) inside every single cell, giving an unparalleled look into the "mind" of the agent.

---

## 🛠️ How It Works

The project is built on three core components that work together.

1.  **The Environment (`env.py`)**: The `GridWorld` class defines the "game." It's a 10x10 grid with rewards: `+10` for the goal, `-10` for obstacles, and `-1` for each step (to encourage efficiency).
2.  **The Agent (`agents.py`)**: The `QAgent` class is the "brain." It uses a **Q-Table** to store the expected future reward for every (state, action) pair. It learns using the **Epsilon-Greedy** policy to balance exploration and exploitation.
3.  **The Training Loop (`main.py`)**: This script orchestrates the entire process. For 15,000 episodes, it:
    * Sets a new random goal and random start.
    * Lets the agent `choose_action()`.
    * Tells the environment to `step()`.
    * Tells the agent to `learn()` from the outcome.
    * Decays `epsilon` over time.

---

## 🏁 How to Run This Project

Follow these steps to get the project running on your machine.

### 1. Create the Project Files

Create a folder for your project (e.g., `Q-Learning-GridWorld`). Inside that folder, create the following files and copy-paste the code from the **"Complete Project Code"** section below.

The final file structure should be:

```bash
Q-Learning-GridWorld/
├── main.py
├── env.py
├── agents.py
├── visualizer.py
├── requirements.txt
└── .gitignore
