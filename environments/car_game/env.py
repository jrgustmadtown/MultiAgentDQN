"""
Car Game - Zero-Sum Two-Player Game Environment

n×n grid (n must be odd) where two cars move simultaneously.
- Each square has points = 2^(Manhattan distance from nearest corner)
- Car A is crasher, Car B is victim
- If cars occupy same space: Car A gets 2×max_points, Car B gets -2×max_points
- Otherwise: Car A gets square points, Car B gets -square points (zero-sum)
"""

import random
import numpy as np


class CarGame(object):
    
    # Define actions (STAY removed - cars must always move)
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, args, current_path):
        """
        Initialize the car game environment.
        
        Args:
            args: Dictionary containing game parameters
            current_path: Path to the current directory
        """
        # Game parameters
        self.num_players = 2  # Two cars
        self.grid_size = args.get('grid_size', 5)
        
        if self.grid_size % 2 == 0:
            raise ValueError("Grid size must be odd!")
        
        # State representation: [car_a_x, car_a_y, car_b_x, car_b_y]
        self.state_size = 4
        
        # Game state variables
        self.car_a_pos = None  # (x, y) position of Car A (crasher)
        self.car_b_pos = None  # (x, y) position of Car B (victim)
        self.done = False
        self.step_count = 0
        self.max_steps = args.get('max_timestep', 100)
        
        # Calculate maximum points (at center)
        center_dist = self.grid_size - 1  # Manhattan distance from corner to center
        self.max_points = 2 ** center_dist
        self.crash_reward = 2 * self.max_points
        self.wall_penalty = -50  # Moderate penalty for hitting walls
        
        # Visualization (optional)
        self.render_flag = args.get('render', False)
        
    def action_space(self):
        """Return the number of possible actions per player."""
        return 4  # UP, DOWN, LEFT, RIGHT (no STAY)
    
    def _calculate_square_points(self, pos):
        """
        Calculate points for a square based on Manhattan distance from nearest corner.
        Points = 2^(distance from nearest corner)
        
        Args:
            pos: (x, y) position tuple
        
        Returns:
            points: Integer points value
        """
        x, y = pos
        n = self.grid_size
        
        # Calculate Manhattan distance to each corner
        dist_to_corners = [
            x + y,                          # Top-left (0, 0)
            x + (n - 1 - y),                # Top-right (0, n-1)
            (n - 1 - x) + y,                # Bottom-left (n-1, 0)
            (n - 1 - x) + (n - 1 - y)       # Bottom-right (n-1, n-1)
        ]
        
        min_dist = min(dist_to_corners)
        points = 2 ** min_dist
        return points
    
    def reset(self):
        """
        Reset the game to initial state with random non-overlapping positions.
        
        Returns:
            state: Initial state [car_a_x, car_a_y, car_b_x, car_b_y]
        """
        self.done = False
        self.step_count = 0
        
        # Generate random non-overlapping starting positions
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(all_positions)
        
        self.car_a_pos = all_positions[0]
        self.car_b_pos = all_positions[1]
        
        # Return the full state
        state = self._get_state()
        return state
    
    def step(self, actions):
        """
        Execute one step of the game with simultaneous moves.
        
        Args:
            actions: List of actions [car_a_action, car_b_action]
        
        Returns:
            next_state: Next state after actions are taken
            reward: Reward for Car A (Car B gets -reward for zero-sum)
            done: Whether the episode is finished
        """
        if len(actions) != 2:
            raise ValueError("Car game requires exactly 2 actions (one per car)")
        
        car_a_action = actions[0]
        car_b_action = actions[1]
        
        self.step_count += 1
        
        # Apply actions and check for wall collisions
        new_car_a_pos, hit_wall_a = self._apply_action_with_collision(self.car_a_pos, car_a_action)
        new_car_b_pos, hit_wall_b = self._apply_action_with_collision(self.car_b_pos, car_b_action)
        
        # Update positions
        self.car_a_pos = new_car_a_pos
        self.car_b_pos = new_car_b_pos
        
        # Check if either car hit a wall
        if hit_wall_a or hit_wall_b:
            # Wall collision - penalty but continue playing
            reward = self.wall_penalty + self._calculate_square_points(self.car_a_pos)
        # Check for crash (same position)
        elif self.car_a_pos == self.car_b_pos:
            # Crash! Car A (crasher) wins big
            reward = self.crash_reward
            self.done = True
        else:
            # Normal move: Car A gets square points
            reward = self._calculate_square_points(self.car_a_pos)
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            self.done = True
        
        next_state = self._get_state()
        
        # Only render if flag is set (don't render every step automatically)
        # Rendering can be called manually from the training script
        
        return next_state, reward, self.done
    
    def _apply_action_with_collision(self, pos, action):
        """
        Apply an action and detect if it would hit a wall.
        
        Args:
            pos: Current (x, y) position
            action: Action index (0-3)
        
        Returns:
            new_pos: New position (stays same if wall hit)
            hit_wall: True if action tried to move into a wall
        """
        x, y = pos
        dx, dy = self.A_DIFF[action]
        
        new_x = x + dx
        new_y = y + dy
        
        # Check if new position is out of bounds
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            return pos, True  # Stay in place, but return wall collision flag
        
        return (new_x, new_y), False
    
    def _apply_action(self, pos, action):
        """
        Apply an action to a position, ensuring it stays within grid bounds.
        
        Args:
            pos: Current (x, y) position
            action: Action index (0-4)
        
        Returns:
            new_pos: New (x, y) position after action
        """
        x, y = pos
        dx, dy = self.A_DIFF[action]
        
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        
        return (new_x, new_y)
    
    def _get_state(self):
        """
        Construct the state representation.
        
        Returns:
            state: Numpy array [car_a_x, car_a_y, car_b_x, car_b_y]
        """
        state = np.array([
            self.car_a_pos[0],
            self.car_a_pos[1],
            self.car_b_pos[0],
            self.car_b_pos[1]
        ])
        return state
    
    def render(self):
        """
        Visualize the current game state with ASCII art.
        """
        print(f"\n{'='*50}")
        print(f"Step {self.step_count}")
        print(f"Car A (Crasher) at {self.car_a_pos}, Car B (Victim) at {self.car_b_pos}")
        print(f"{'='*50}")
        
        # Create grid visualization
        for x in range(self.grid_size):
            row = []
            for y in range(self.grid_size):
                pos = (x, y)
                points = self._calculate_square_points(pos)
                
                if pos == self.car_a_pos and pos == self.car_b_pos:
                    cell = "XX"  # Crash!
                elif pos == self.car_a_pos:
                    cell = " A"
                elif pos == self.car_b_pos:
                    cell = " B"
                else:
                    cell = f"{points:2d}"
                row.append(cell)
            print(" | ".join(row))
        
        print(f"{'='*50}\n")
    
    def close(self):
        """Clean up resources."""
        pass
