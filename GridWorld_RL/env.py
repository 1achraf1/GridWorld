import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from GridWorld_RL.agents import Agent,QAgent

class GridWorld:
    def __init__(self, height=5, width=5, start=None, goal=None,
                 reward_goal=10, reward_step=-1, obstacles=None, obstacle_penalty=-10,
                 stochastic=True, moving_goal=False, moving_obstacles=False,
                 goal_move_frequency=5, obstacle_move_frequency=3):
        """
        Initialize the GridWorld environment.

        Args:
            height (int): Height of the grid.
            width (int): Width of the grid.
            start (tuple): Starting position (row, col). If None, randomly selected.
            goal (tuple or list): Goal position(s). If None, randomly selected.
            reward_goal (float): Reward for reaching the goal.
            reward_step (float): Reward for each step.
            obstacles (list): List of obstacle positions [(row, col), ...]. If None, random number (1-10) of obstacles created.
            obstacle_penalty (float): Penalty for hitting obstacles/walls.
            stochastic (bool): If True, actions have 80% success rate with 10% drift.
                              If False, actions are deterministic (100% success rate).
            moving_goal (bool): If True, goal moves to a new random position each episode.
            moving_obstacles (bool): If True, obstacles move during the episode.
            goal_move_frequency (int): Steps between goal movements (if moving_goal=True during episode).
            obstacle_move_frequency (int): Steps between obstacle movements (if moving_obstacles=True).
        """
        self.height, self.width = height, width
        self.stochastic = stochastic
        self.moving_goal = moving_goal
        self.moving_obstacles = moving_obstacles
        self.goal_move_frequency = goal_move_frequency
        self.obstacle_move_frequency = obstacle_move_frequency

        # Set start first
        if start is not None:
            self.start = tuple(start)
        else:
            # Random start position
            self.start = (np.random.randint(0, self.height), np.random.randint(0, self.width))

        # Set obstacles 
        if obstacles is not None:
            self.initial_obstacles = [tuple(o) for o in obstacles]
            self.obstacles = set(tuple(o) for o in obstacles)
        else:
            # Random number of obstacles between 1 and 10
            num_obstacles = np.random.randint(1, 11)
            self.initial_obstacles = []
            self.obstacles = set()

            for _ in range(num_obstacles):
                obs = self._random_free_cell(exclude=[self.start] + self.initial_obstacles)
                self.initial_obstacles.append(obs)
                self.obstacles.add(obs)

        # Set goals
        if goal is None:
            self.initial_goals = [self._random_free_cell(exclude=[self.start] + list(self.obstacles))]
        elif isinstance(goal, list):
            self.initial_goals = [tuple(g) for g in goal]
        else:
            self.initial_goals = [tuple(goal)]

        self.goals = self.initial_goals.copy()

        # Check for conflicts
        assert self.start not in self.obstacles, "Start position can't be an obstacle"
        for g in self.goals:
            assert g not in self.obstacles, f"Goal {g} can't be an obstacle"

        # Agent
        from Agents import Agent
        self.agent = Agent()
        self.agent.set_position(self.start)

        # Rewards
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.obstacle_penalty = obstacle_penalty
        self._steps = 0

        # Colormap: 0 empty, 2 agent, 3 goal, 4 obstacle
        self.cmap = colors.ListedColormap(["white", "blue", "green", "black"])
        self.norm = colors.BoundaryNorm([0, 2, 3, 4, 5], self.cmap.N)

    def _random_free_cell(self, exclude=[]):
        """Find a random cell not in exclude or obstacles."""
        exclude_set = set(exclude) | self.obstacles if hasattr(self, 'obstacles') else set(exclude)
        max_attempts = self.height * self.width

        for _ in range(max_attempts):
            cell = (np.random.randint(0, self.height), np.random.randint(0, self.width))
            if cell not in exclude_set:
                return cell

        # Fallback: return any free cell
        for r in range(self.height):
            for c in range(self.width):
                cell = (r, c)
                if cell not in exclude_set:
                    return cell

        return (0, 0)  # Last resort

    def _move_goal(self):
        """Move goal to a new random position."""
        if not self.moving_goal:
            return

        # Exclude current agent position, obstacles, and current goals
        exclude = [self.agent.pos] + list(self.obstacles) + self.goals
        new_goal = self._random_free_cell(exclude=exclude)

        print(f"  [Goal moved from {self.goals[0]} to {new_goal}]")
        self.goals = [new_goal]

    def _move_obstacles(self):
        """Move obstacles to adjacent cells."""
        if not self.moving_obstacles or not self.obstacles:
            return

        new_obstacles = set()

        for obs in self.obstacles:
            # Get possible adjacent cells (up, right, down, left)
            adjacent = [
                (obs[0] - 1, obs[1]),  # up
                (obs[0], obs[1] + 1),  # right
                (obs[0] + 1, obs[1]),  # down
                (obs[0], obs[1] - 1)  # left
            ]

            # Filter valid moves (within bounds, not occupied)
            valid_moves = []
            for new_pos in adjacent:
                r, c = new_pos
                if (0 <= r < self.height and 0 <= c < self.width and
                        new_pos != self.agent.pos and
                        new_pos not in self.goals and
                        new_pos not in new_obstacles):
                    valid_moves.append(new_pos)

            # Choose random valid move or stay in place
            if valid_moves and np.random.random() < 0.7:  # 70% chance to move
                new_obstacles.add(valid_moves[np.random.choice(len(valid_moves))])
            else:
                new_obstacles.add(obs)  # Stay in place

        self.obstacles = new_obstacles

    def reset(self):
        """Reset the environment for a new episode."""
        self.agent.reset()
        self._steps = 0

        # Reset obstacles to initial positions
        self.obstacles = set(self.initial_obstacles)

        # Move goal to new position if moving_goal is True
        if self.moving_goal:
            exclude = [self.start] + list(self.obstacles)
            new_goal = self._random_free_cell(exclude=exclude)
            self.goals = [new_goal]
            print(f"New episode: Goal set to {new_goal}")
        else:
            # Reset to initial goal positions
            self.goals = self.initial_goals.copy()

        return self._get_obs()

    def step(self, action):
        self._steps += 1

        # Move obstacles if it's time
        if self.moving_obstacles and self._steps % self.obstacle_move_frequency == 0:
            self._move_obstacles()

        # Determine actual action based on stochastic setting
        if self.stochastic:
            # Stochastic: 80% intended action, 10% left drift, 10% right drift
            actions = [0, 1, 2, 3]

            if action == 0:  # Up
                probabilities = [0.8, 0.1, 0, 0.1]
                actual_action = np.random.choice(actions, p=probabilities)
            elif action == 1:  # Right
                probabilities = [0.1, 0.8, 0.1, 0]
                actual_action = np.random.choice(actions, p=probabilities)
            elif action == 2:  # Down
                probabilities = [0, 0.1, 0.8, 0.1]
                actual_action = np.random.choice(actions, p=probabilities)
            elif action == 3:  # Left
                probabilities = [0.1, 0, 0.1, 0.8]
                actual_action = np.random.choice(actions, p=probabilities)
            else:
                actual_action = action  # Should not happen with current action space
        else:
            # Deterministic: action is executed exactly as intended
            actual_action = action

        proposed_pos = self.agent.move(actual_action)
        proposed_r, proposed_c = proposed_pos

        # --- Part 1: Movement Logic ---
        # First, let's check if the move is VALID.
        is_valid = (0 <= proposed_r < self.height and
                    0 <= proposed_c < self.width and
                    proposed_pos not in self.obstacles)

        if is_valid:
            self.agent.pos = proposed_pos
        else:
            pass  # Agent stays in place

        # Dealing with reward
        if not is_valid:
            reward = self.obstacle_penalty
            done = False
        elif self.agent.pos in self.goals:
            reward = self.reward_goal
            done = True
        else:
            reward = self.reward_step
            done = False

        return self._get_obs(), reward, done, actual_action

    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        # mark obstacles
        for (orow, ocol) in self.obstacles:
            grid[orow, ocol] = 4
        # mark goals
        for (gr, gc) in self.goals:
            grid[gr, gc] = 3
        # mark agent
        ar, ac = self.agent.pos
        grid[ar, ac] = 2

        return grid

    def set_goal(self, new_goal):
        """Update the goal position during training."""
        new_goal = tuple(new_goal)

        # Make sure new goal is valid
        assert new_goal not in self.obstacles, "Goal cannot be on an obstacle"
        assert new_goal != self.start, "Goal cannot be at the start position"

        self.goals = [new_goal]

    def render(self):
        grid = self._get_obs()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap=self.cmap, norm=self.norm, extent=[0, self.width, self.height, 0])
        # grid lines
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(color="black", linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Show configuration in title
        mode = "Stochastic" if self.stochastic else "Deterministic"
        dynamics = []
        if self.moving_goal:
            dynamics.append("Moving Goal")
        if self.moving_obstacles:
            dynamics.append("Moving Obstacles")
        dynamics_str = " | " + ", ".join(dynamics) if dynamics else ""

        plt.title(f"Steps: {self._steps} | Mode: {mode}{dynamics_str}")
        plt.show()

    def _get_obs_from_pos(self, agent_pos):
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        for (orow, ocol) in self.obstacles:
            grid[orow, ocol] = 4
        for (gr, gc) in self.goals:
            grid[gr, gc] = 3
        ar, ac = agent_pos
        grid[ar, ac] = 2
        return grid
    
