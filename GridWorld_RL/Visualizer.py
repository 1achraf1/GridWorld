
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from IPython.display import HTML
from gridworld_rl.env import GridWorld
from gridworld_rl.agents import Agent, QAgent


class GridVisualizer:
    """A class to handle all visualization for the GridWorld."""

    def __init__(self, grid_world):
        self.world = grid_world
        # Colormap: 0=empty, 1=unused, 2=agent, 3=goal, 4=obstacle
        self.cmap = colors.ListedColormap(["white", "gray", "blue", "green", "black"])
        self.norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], self.cmap.N)

    def _prepare_plot(self):
        """Creates the figure, axis, and static background for visualizations."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the static grid background with clear squares
        grid_bg = np.zeros((self.world.height, self.world.width))

        # Draw empty cells with a subtle gradient effect
        for r in range(self.world.height):
            for c in range(self.world.width):
                rect = Rectangle((c, r), 1, 1, linewidth=2.5,
                                 edgecolor='#2c3e50', facecolor='#ecf0f1')
                ax.add_patch(rect)

        # Draw obstacles as dark gray squares with texture
        for obs in self.world.obstacles:
            rect = Rectangle((obs[1], obs[0]), 1, 1, linewidth=2.5,
                             edgecolor='#2c3e50', facecolor='#34495e')
            ax.add_patch(rect)
            # Add X pattern to obstacles
            ax.plot([obs[1] + 0.2, obs[1] + 0.8], [obs[0] + 0.2, obs[0] + 0.8],
                    'k-', linewidth=2, alpha=0.3)
            ax.plot([obs[1] + 0.2, obs[1] + 0.8], [obs[0] + 0.8, obs[0] + 0.2],
                    'k-', linewidth=2, alpha=0.3)

        # Draw goals as vibrant green squares with glow effect
        for goal in self.world.goals:
            # Outer glow
            glow = Rectangle((goal[1] - 0.05, goal[0] - 0.05), 1.1, 1.1, linewidth=0,
                             facecolor='#2ecc71', alpha=0.3, zorder=1)
            ax.add_patch(glow)
            # Main goal square
            rect = Rectangle((goal[1], goal[0]), 1, 1, linewidth=2.5,
                             edgecolor='#27ae60', facecolor='#2ecc71', zorder=2)
            ax.add_patch(rect)
            # Add a large star marker in the goal
            ax.text(goal[1] + 0.5, goal[0] + 0.5, '*',
                    fontsize=50, ha='center', va='center',
                    color='#f39c12', fontweight='bold', zorder=3)

        # Set axis limits and remove ticks
        ax.set_xlim(0, self.world.width)
        ax.set_ylim(0, self.world.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # To match array indexing (0,0 at top-left)
        ax.set_xticks(range(self.world.width + 1))
        ax.set_yticks(range(self.world.height + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_facecolor('#bdc3c7')

        return fig, ax

    def render_policy(self, policy):
        """Renders the grid world with arrows indicating the policy."""
        fig, ax = self._prepare_plot()

        action_arrows = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}
        action_colors = {0: '#e74c3c', 1: '#3498db', 2: '#9b59b6', 3: '#f39c12'}

        for r in range(self.world.height):
            for c in range(self.world.width):
                state = (r, c)
                if state in self.world.obstacles or state in self.world.goals:
                    continue

                action = policy[r, c]
                dx, dy = action_arrows.get(action, (0, 0))
                color = action_colors.get(action, 'black')
                # Arrow is centered at (c+0.5, r+0.5) - center of square
                ax.arrow(c + 0.5, r + 0.5, dx, dy,
                         head_width=0.25, head_length=0.2, fc=color, ec=color,
                         linewidth=2.5, zorder=5)

        ax.set_title("Agent's Learned Policy", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def run_and_animate_episode(self, agent, episode_num=1, jupyter=False):
        """
        Runs a single episode with a trained agent and creates a smooth animation.

        Args:
            agent (Agents): The trained Q-learning agent.
            episode_num (int): The episode number to display in the title.
            jupyter (bool): If True, returns HTML for Jupyter. If False, shows in window.
        """
        fig, ax = self._prepare_plot()

        # --- Run the episode and record history ---
        history = []
        self.world.reset()  # Reset the environment
        pos = self.world.agent.pos  # Get agent position after reset
        done = False
        cumulative_reward = 0

        while not done:
            state = pos[0] * self.world.width + pos[1]
            # Exploit the learned policy (no random moves)
            action = np.argmax(agent.q_table[state])

            obs, reward, done, actual_action = self.world.step(action)
            cumulative_reward += reward
            next_pos = self.world.agent.pos  # Get the updated agent position

            history.append({
                'pos': pos,
                'reward': cumulative_reward,
                'done': done
            })

            pos = next_pos

            # Add final position when goal is reached
            if done:
                history.append({
                    'pos': pos,
                    'reward': cumulative_reward,
                    'done': done
                })
                break

            # Safety break for policies that lead to loops
            if len(history) > self.world.height * self.world.width * 2:
                print("Animation stopped: Episode seems to be in an infinite loop.")
                break

        # --- Create the agent as a circle clearly inside the square ---
        agent_circle = Circle((0.5, 0.5), 0.4, fc='#3498db', ec='#2c3e50',
                              linewidth=3, zorder=10)
        # Agent inner detail
        agent_eye1 = Circle((0.4, 0.4), 0.08, fc='white', zorder=11)
        agent_eye2 = Circle((0.6, 0.4), 0.08, fc='white', zorder=11)

        title_text = ax.set_title(f"Episode: {episode_num} | Step: 0 | Reward: 0.0",
                                  fontsize=16, fontweight='bold', pad=20)

        def init():
            ax.add_patch(agent_circle)
            ax.add_patch(agent_eye1)
            ax.add_patch(agent_eye2)
            return agent_circle, agent_eye1, agent_eye2, title_text

        def update(frame_num):
            data = history[frame_num]
            pos = data['pos']
            reward = data['reward']
            done = data['done']

            # Update agent position - center it in the square
            center_x = pos[1] + 0.5
            center_y = pos[0] + 0.5
            agent_circle.center = (center_x, center_y)
            agent_eye1.center = (center_x - 0.1, center_y - 0.1)
            agent_eye2.center = (center_x + 0.1, center_y - 0.1)

            # Change color when reaching goal
            if done:
                agent_circle.set_facecolor('#2ecc71')
                title_text.set_text(f"GOAL REACHED! | Steps: {frame_num} | Reward: {reward:.1f}")
                title_text.set_color('#27ae60')
            else:
                title_text.set_text(f"Episode: {episode_num} | Step: {frame_num} | Reward: {reward:.1f}")

            return agent_circle, agent_eye1, agent_eye2, title_text

        # Create and display the animation
        anim = FuncAnimation(fig, update, frames=len(history),
                             init_func=init, blit=True, interval=300, repeat=False)

        plt.tight_layout()

        if jupyter:
            # For Jupyter notebooks - return HTML
            html = HTML(anim.to_jshtml())
            plt.close(fig)
            return html
        else:
            # For regular Python scripts - show in window
            plt.show()
            return anim

    def test_agent_live(self, agent, max_steps=100, speed=0.4):
        """
        Opens a matplotlib window showing the agent following its learned policy in real-time.

        Args:
            agent (Agents): The trained Q-learning agent.
            max_steps (int): Maximum steps to prevent infinite loops.
            speed (float): Time between steps in seconds (lower = faster).
        """
        fig, ax = self._prepare_plot()

        # Create agent as a blue circle with eyes
        agent_circle = Circle((0.5, 0.5), 0.4, fc='#3498db', ec='#2c3e50',
                              linewidth=3, zorder=10)
        agent_eye1 = Circle((0.4, 0.4), 0.08, fc='white', zorder=11)
        agent_eye2 = Circle((0.6, 0.4), 0.08, fc='white', zorder=11)

        ax.add_patch(agent_circle)
        ax.add_patch(agent_eye1)
        ax.add_patch(agent_eye2)

        # Add trail effect
        trail_circles = []

        # Initialize episode
        self.world.reset()
        pos = self.world.agent.pos
        agent_circle.center = (pos[1] + 0.5, pos[0] + 0.5)
        agent_eye1.center = (pos[1] + 0.4, pos[0] + 0.4)
        agent_eye2.center = (pos[1] + 0.6, pos[0] + 0.4)

        step = 0
        cumulative_reward = 0
        done = False

        title_text = ax.set_title(f"Testing Agent | Step: {step} | Reward: {cumulative_reward:.1f}",
                                  fontsize=16, fontweight='bold', pad=20,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.ion()  # Turn on interactive mode
        plt.show()

        while not done and step < max_steps:
            plt.pause(speed)  # Pause to see each step clearly

            # Add trail at current position
            if len(trail_circles) < 5:  # Keep last 5 positions
                trail = Circle((pos[1] + 0.5, pos[0] + 0.5), 0.15,
                               fc='#3498db', alpha=0.3 - len(trail_circles) * 0.05, zorder=5)
                ax.add_patch(trail)
                trail_circles.append(trail)
            else:
                # Remove oldest trail
                trail_circles[0].remove()
                trail_circles.pop(0)
                trail = Circle((pos[1] + 0.5, pos[0] + 0.5), 0.15,
                               fc='#3498db', alpha=0.3, zorder=5)
                ax.add_patch(trail)
                trail_circles.append(trail)

            state = pos[0] * self.world.width + pos[1]
            action = np.argmax(agent.q_table[state])

            obs, reward, done, actual_action = self.world.step(action)
            cumulative_reward += reward
            step += 1

            # Get the updated agent position
            next_pos = self.world.agent.pos

            # Update agent position
            center_x = next_pos[1] + 0.5
            center_y = next_pos[0] + 0.5
            agent_circle.center = (center_x, center_y)
            agent_eye1.center = (center_x - 0.1, center_y - 0.1)
            agent_eye2.center = (center_x + 0.1, center_y - 0.1)

            title_text.set_text(f"Testing Agent | Step: {step} | Reward: {cumulative_reward:.1f}")

            fig.canvas.draw()
            fig.canvas.flush_events()

            pos = next_pos

        # Goal reached animation
        if done:
            # Flash effect
            for i in range(3):
                agent_circle.set_facecolor('#f39c12')
                title_text.set_text(f"GOAL REACHED! | Steps: {step} | Final Reward: {cumulative_reward:.1f}")
                title_text.set_color('#27ae60')
                title_text.set_bbox(dict(boxstyle='round', facecolor='#2ecc71', alpha=0.8))
                fig.canvas.draw()
                plt.pause(0.15)

                agent_circle.set_facecolor('#2ecc71')
                fig.canvas.draw()
                plt.pause(0.15)

            # Final celebration
            agent_circle.set_facecolor('#2ecc71')
            agent_circle.set_radius(0.45)
        else:
            agent_circle.set_facecolor('#e74c3c')
            title_text.set_text(f"Max steps reached | Steps: {step} | Reward: {cumulative_reward:.1f}")
            title_text.set_color('#c0392b')
            title_text.set_bbox(dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))

        fig.canvas.draw()
        plt.ioff()  # Turn off interactive mode
        plt.show()

        # Print summary
        if done:
            print(f"\nEpisode finished: SUCCESS!")
            print(f"Steps taken: {step}")
            print(f"Final reward: {cumulative_reward:.1f}")
        else:
            print(f"\nEpisode finished: TIMEOUT")
            print(f"Steps taken: {step}")
            print(f"Final reward: {cumulative_reward:.1f}")

    def plot_q_value_heatmap(self, agent, action=None):
        """
        Plots a heatmap of Q-values for the grid world.

        Args:
            agent (Agents): The trained Q-learning agent.
            action (int, optional): If specified, shows Q-values for that action only.
                                   If None, shows all 4 actions in subplots.
                                   0=Up, 1=Right, 2=Down, 3=Left
        """
        action_names = ['Up ^', 'Right >', 'Down v', 'Left <']

        if action is not None:
            # Single action heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Reshape Q-values for this action into grid
            q_grid = agent.q_table[:, action].reshape((self.world.height, self.world.width))

            # Create heatmap
            im = ax.imshow(q_grid, cmap='RdYlGn', aspect='auto')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Q-Value', rotation=270, labelpad=20, fontsize=12)

            # Add grid lines
            ax.set_xticks(np.arange(self.world.width))
            ax.set_yticks(np.arange(self.world.height))
            ax.set_xticks(np.arange(-0.5, self.world.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.world.height, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            ax.tick_params(which='minor', size=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Add Q-values as text in each cell
            for r in range(self.world.height):
                for c in range(self.world.width):
                    state = (r, c)
                    if state in self.world.obstacles:
                        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               facecolor='gray', alpha=0.5))
                        ax.text(c, r, 'X', ha='center', va='center',
                                fontsize=20, color='black', fontweight='bold')
                    elif state in self.world.goals:
                        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               facecolor='gold', alpha=0.3))
                        ax.text(c, r, f'{q_grid[r, c]:.2f}\n*',
                                ha='center', va='center', fontsize=10, fontweight='bold')
                    else:
                        ax.text(c, r, f'{q_grid[r, c]:.2f}',
                                ha='center', va='center', fontsize=11, fontweight='bold')

            ax.set_title(f'Q-Values for Action: {action_names[action]}',
                         fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.show()

        else:
            # All 4 actions in subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            for act in range(4):
                ax = axes[act]

                # Reshape Q-values for this action into grid
                q_grid = agent.q_table[:, act].reshape((self.world.height, self.world.width))

                # Create heatmap
                im = ax.imshow(q_grid, cmap='RdYlGn', aspect='auto')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Q-Value', rotation=270, labelpad=15, fontsize=10)

                # Add grid lines
                ax.set_xticks(np.arange(self.world.width))
                ax.set_yticks(np.arange(self.world.height))
                ax.set_xticks(np.arange(-0.5, self.world.width, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, self.world.height, 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
                ax.tick_params(which='minor', size=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Add Q-values as text in each cell
                for r in range(self.world.height):
                    for c in range(self.world.width):
                        state = (r, c)
                        if state in self.world.obstacles:
                            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                   facecolor='gray', alpha=0.5))
                            ax.text(c, r, 'X', ha='center', va='center',
                                    fontsize=16, color='black', fontweight='bold')
                        elif state in self.world.goals:
                            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                   facecolor='gold', alpha=0.3))
                            ax.text(c, r, f'{q_grid[r, c]:.1f}\n*',
                                    ha='center', va='center', fontsize=9, fontweight='bold')
                        else:
                            ax.text(c, r, f'{q_grid[r, c]:.2f}',
                                    ha='center', va='center', fontsize=9, fontweight='bold')

                ax.set_title(f'{action_names[act]}', fontsize=12, fontweight='bold', pad=10)

            plt.suptitle('Q-Value Heatmaps for All Actions',
                         fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()

    def plot_q_values_in_cells(self, agent):
        """
        Plots all 4 Q-values inside each grid cell in a single view.
        Each cell shows: Up (top), Right (right), Down (bottom), Left (left)
        """
        fig, ax = self._prepare_plot()

        # Action positions within each cell (relative to center)
        action_positions = {
            0: (0, -0.25),  # Up - top
            1: (0.25, 0),  # Right - right
            2: (0, 0.25),  # Down - bottom
            3: (-0.25, 0)  # Left - left
        }

        action_colors = {
            0: '#e74c3c',  # Up - red
            1: '#3498db',  # Right - blue
            2: '#9b59b6',  # Down - purple
            3: '#f39c12'  # Left - orange
        }

        # Get max and min Q-values for color normalization
        q_max = agent.q_table.max()
        q_min = agent.q_table.min()
        q_range = q_max - q_min if q_max != q_min else 1

        for r in range(self.world.height):
            for c in range(self.world.width):
                state = (r, c)

                # Skip obstacles and goals
                if state in self.world.obstacles or state in self.world.goals:
                    continue

                # Get state index
                state_idx = r * self.world.width + c

                # Plot each Q-value in its position
                for action in range(4):
                    q_value = agent.q_table[state_idx, action]
                    dx, dy = action_positions[action]

                    # Normalize Q-value for alpha (transparency)
                    alpha = 0.3 + 0.7 * ((q_value - q_min) / q_range)

                    # Plot Q-value text
                    ax.text(c + 0.5 + dx, r + 0.5 + dy,
                            f'{q_value:.1f}',
                            ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color=action_colors[action],
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor=action_colors[action],
                                      alpha=alpha, linewidth=1.5))

        # Add legend
        legend_elements = [
            Patch(facecolor=action_colors[0], label='Up ^'),
            Patch(facecolor=action_colors[1], label='Right >'),
            Patch(facecolor=action_colors[2], label='Down v'),
            Patch(facecolor=action_colors[3], label='Left <')
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(1.02, 1), fontsize=10)

        ax.set_title('Q-Values for All Actions (in each cell)',
                     fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
