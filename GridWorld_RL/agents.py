import numpy as np

class Agent:
    def __init__(self):
        self.start = None
        self.pos = None

    def set_position(self, start):
        self.start = tuple(start)

    def reset(self):
        self.pos = tuple(self.start)

    def move(self, action):
        r, c = self.pos

        if action == 0:      # up
            r -= 1
        elif action == 1:    # right
            c += 1
        elif action == 2:    # down
            r += 1
        elif action == 3:    # left
            c -= 1

        
        pos = (r, c)

        return pos

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initializes the Q-learning agent.

        Args:
            state_size (int): The number of states in the environment (e.g., 25 for a 5x5 grid).
            action_size (int): The number of possible actions (e.g., 4 for up, right, down, left).
            learning_rate (float): Alpha, controls how much we update Q-values.
            discount_factor (float): Gamma, weights future rewards.
            epsilon (float): The initial probability of taking a random action.
            epsilon_decay (float): Multiplier to decrease epsilon after each episode.
            min_epsilon (float): The lowest value epsilon is allowed to reach.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize the Q-table with all zeros
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self,state):
      if np.random.rand() <self.epsilon:
        return np.random.choice(self.action_size)
      else:
        return np.argmax(self.q_table[state])

    

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Q-learning formula.

        Args:
            state (int): The starting state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The resulting state.
            done (bool): True if the episode has terminated.
        """
        

        q_next = np.max(self.q_table[next_state]) if not done else 0
        self.q_table[state,action] = self.q_table[state,action] + self.lr * (reward + self.gamma * q_next - self.q_table[state,action])

    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
