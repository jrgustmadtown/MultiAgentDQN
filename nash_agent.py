"""
Nash-Q Agent - Uses joint Q-values for multi-agent learning.

Key difference from DQN Agent:
- Single shared Q-function outputting Q(s, a_A, a_B) for all action pairs
- Will use Nash equilibrium computation for action selection (Phase 2-4)
"""

import random
import numpy as np
from nash_brain import NashBrain
from nash_solver import compute_minimax_value, compute_nash_equilibrium


class NashAgent(object):
    """Agent for Nash-Q Learning."""

    def __init__(self, state_size, num_actions_per_agent, num_agents, arguments, agent_id=0, weights_path=''):
        self.state_size = state_size
        self.num_actions_per_agent = num_actions_per_agent
        self.num_agents = num_agents
        self.joint_action_size = num_actions_per_agent ** num_agents  # 4^2 = 16
        
        self.agent_id = agent_id  # For identification
        
        # Exploration parameters
        self.epsilon = arguments['epsilon_start']
        self.epsilon_final = arguments['epsilon_stop']
        self.epsilon_decay = arguments['epsilon_decay']
        
        # Learning parameters
        self.memory_capacity = arguments['memory_capacity']
        self.batch_size = arguments['batch_size']
        self.gamma = arguments['gamma']
        self.update_target_frequency = arguments['update_target_frequency']
        self.double_dqn = arguments['double_dqn']
        self.test = arguments['test']
        self.experience_replay_type = arguments['experience_replay']
        self.pr_scale = arguments.get('pr_scale', 0.5)  # For PER
        self.train_frequency = arguments.get('train_frequency', 1)  # Train every N steps
        self.reward_scale = arguments.get('reward_scale', 1.0)  # Scale rewards down
        
        # Experience replay
        if self.experience_replay_type == 'UER':
            from uniform_experience_replay import Memory
            self.memory = Memory(self.memory_capacity)
        elif self.experience_replay_type == 'PER':
            from prioritized_experience_replay import Memory
            self.memory = Memory(self.memory_capacity, self.pr_scale)
        else:
            raise ValueError("Invalid experience replay type")
        
        # Nash-Q brain
        import os
        brain_name = os.path.join(weights_path, f'nash_agent_{agent_id}.pth')
        self.brain = NashBrain(state_size, num_actions_per_agent, num_agents, brain_name, arguments)
        
        # Track losses
        self.losses = []
        self.episode_losses = []  # Per-episode tracking
        self.max_loss_history = 10000  # Limit loss history to prevent memory issues
        
        # Step counter
        self.steps = 0

    def greedy_actor(self, state):
        """
        Phase 4: Use Nash equilibrium policy with wall-aware action masking.
        
        Mask invalid actions (hitting walls) in Q-matrix before computing Nash equilibrium.
        """
        if np.random.rand() <= self.epsilon:
            # Random exploration with valid actions only
            action_A = self._get_random_valid_action(state, agent_index=0)
            action_B = self._get_random_valid_action(state, agent_index=1)
        else:
            # Nash equilibrium exploitation with masked Q-values
            q_matrix = self.brain.get_q_matrix(state, target=False)
            
            # Mask invalid actions with very negative values
            q_matrix_masked = self._mask_invalid_actions(state, q_matrix)
            
            nash_value, action_A, action_B = compute_nash_equilibrium(q_matrix_masked)
        
        return action_A, action_B
    
    def _mask_invalid_actions(self, state, q_matrix, grid_size=5):
        """
        Mask Q-values for invalid actions (hitting walls) with -inf.
        
        For Nash equilibrium:
        - Car A (maximizer) won't pick actions with -inf rows
        - Car B (minimizer) won't pick actions with -inf columns
        
        Args:
            state: [car_a_x, car_a_y, car_b_x, car_b_y]
            q_matrix: [4, 4] Q-values for all joint actions
            
        Returns:
            q_matrix_masked: Q-matrix with invalid actions set to -inf
        """
        car_a_x, car_a_y = int(state[0]), int(state[1])
        car_b_x, car_b_y = int(state[2]), int(state[3])
        
        q_matrix_masked = q_matrix.copy()
        
        # Mask invalid actions for Car A (rows)
        # UP=0, DOWN=1, LEFT=2, RIGHT=3
        if car_a_x == 0:  # At top, can't go UP
            q_matrix_masked[0, :] = -np.inf
        if car_a_x == grid_size - 1:  # At bottom, can't go DOWN
            q_matrix_masked[1, :] = -np.inf
        if car_a_y == 0:  # At left, can't go LEFT
            q_matrix_masked[2, :] = -np.inf
        if car_a_y == grid_size - 1:  # At right, can't go RIGHT
            q_matrix_masked[3, :] = -np.inf
        
        # Mask invalid actions for Car B (columns)
        if car_b_x == 0:  # At top, can't go UP
            q_matrix_masked[:, 0] = -np.inf
        if car_b_x == grid_size - 1:  # At bottom, can't go DOWN
            q_matrix_masked[:, 1] = -np.inf
        if car_b_y == 0:  # At left, can't go LEFT
            q_matrix_masked[:, 2] = -np.inf
        if car_b_y == grid_size - 1:  # At right, can't go RIGHT
            q_matrix_masked[:, 3] = -np.inf
        
        return q_matrix_masked
    
    def _get_random_valid_action(self, state, agent_index, grid_size=5):
        """Get a random action that won't hit a wall."""
        if agent_index == 0:
            x, y = int(state[0]), int(state[1])
        else:
            x, y = int(state[2]), int(state[3])
        
        # Check which actions are valid (won't hit walls)
        # UP=0, DOWN=1, LEFT=2, RIGHT=3
        valid_actions = []
        if x > 0:  # Can go UP
            valid_actions.append(0)
        if x < grid_size - 1:  # Can go DOWN
            valid_actions.append(1)
        if y > 0:  # Can go LEFT
            valid_actions.append(2)
        if y < grid_size - 1:  # Can go RIGHT
            valid_actions.append(3)
        
        if len(valid_actions) == 0:
            # Should never happen, but fallback
            return random.randrange(self.num_actions_per_agent)
        
        return random.choice(valid_actions)
    
    def joint_action_to_index(self, action_A, action_B):
        """Convert (action_A, action_B) to joint action index."""
        return action_A * self.num_actions_per_agent + action_B
    
    def index_to_joint_action(self, joint_index):
        """Convert joint action index to (action_A, action_B)."""
        action_A = joint_index // self.num_actions_per_agent
        action_B = joint_index % self.num_actions_per_agent
        return action_A, action_B

    def observe(self, state, action_A, action_B, reward, new_state, done):
        """Store experience in memory."""
        # Scale reward to reduce Q-value magnitude
        scaled_reward = reward * self.reward_scale
        
        # Package experience as tuple
        sample = (state, action_A, action_B, scaled_reward, new_state, done)
        
        if self.experience_replay_type == 'PER':
            # Compute TD error for prioritization
            q_values = self.brain.predict_one_sample(state)
            q_values_new = self.brain.predict_one_sample(new_state, target=True)
            
            joint_index = self.joint_action_to_index(action_A, action_B)
            
            if done:
                target = scaled_reward
            else:
                target = scaled_reward + self.gamma * np.max(q_values_new)
            
            error = abs(target - q_values[joint_index])
            self.memory.remember(sample, error)
        else:
            # UER just stores the sample
            self.memory.remember(sample)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
    
    def step_scheduler(self):
        """Step the learning rate scheduler (call once per episode)."""
        self.brain.step_scheduler()

    def replay(self):
        """Train on a batch of experiences."""
        # Increment step counter FIRST
        self.steps += 1
        
        # Only train every N steps
        if self.steps % self.train_frequency != 0:
            return
            
        # Sample batch
        try:
            if self.experience_replay_type == 'UER':
                samples = self.memory.sample(self.batch_size)
                states = np.array([s[0] for s in samples])
                actions_A = np.array([s[1] for s in samples])
                actions_B = np.array([s[2] for s in samples])
                rewards = np.array([s[3] for s in samples])
                new_states = np.array([s[4] for s in samples])
                dones = np.array([s[5] for s in samples])
                sample_weight = None
                batch_indices = None
            elif self.experience_replay_type == 'PER':
                sample_batch, batch_indices, batch_priorities = self.memory.sample(self.batch_size)
                states = np.array([s[1][0] for s in sample_batch])
                actions_A = np.array([s[1][1] for s in sample_batch])
                actions_B = np.array([s[1][2] for s in sample_batch])
                rewards = np.array([s[1][3] for s in sample_batch])
                new_states = np.array([s[1][4] for s in sample_batch])
                dones = np.array([s[1][5] for s in sample_batch])
                
                # Compute importance sampling weights
                total = self.memory.memory.total()
                sample_weight = [(total / batch_priorities[i]) ** 0.4 for i in range(len(sample_batch))]
                sample_weight = np.array(sample_weight) / max(sample_weight)
        except:
            # Not enough samples yet
            return
        # Compute targets
        targets = self.find_targets(states, actions_A, actions_B, rewards, new_states, dones)
        
        # Train
        self.brain.train(states, targets, sample_weight=sample_weight, epochs=1, verbose=0)
        
        # Track loss (with bounded history)
        loss_value = self.brain.last_loss
        self.losses.append(loss_value)
        self.episode_losses.append(loss_value)
        
        # Keep losses list bounded to prevent memory issues
        if len(self.losses) > self.max_loss_history:
            self.losses = self.losses[-self.max_loss_history:]
        
        # Update PER priorities
        if self.experience_replay_type == 'PER':
            predictions = self.brain.predict(states, target=False)
            td_errors = []
            for i in range(len(states)):
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                td_error = abs(targets[i][joint_index] - predictions[i][joint_index])
                td_errors.append(td_error)
            self.memory.update(batch_indices, td_errors)
        
        # Update target network (steps already incremented at start)
        if self.steps % self.update_target_frequency == 0:
            self.brain.update_target_model()

    def find_targets(self, states, actions_A, actions_B, rewards, new_states, dones):
        """
        Phase 3: Nash-Q Learning - Use Nash equilibrium value instead of max Q.
        
        Bellman equation: Q(s, a_A, a_B) = r + γ × V*(s')
        where V*(s') = Nash equilibrium value = minimax(Q(s'))
        """
        batch_size = len(states)
        targets = self.brain.predict(states, target=False)
        
        if self.double_dqn:
            # Double DQN with Nash equilibrium
            q_values_new_state_target = self.brain.predict(new_states, target=True)
            
            for i in range(batch_size):
                if dones[i]:
                    target_value = rewards[i]
                else:
                    # NASH-Q: Compute Nash equilibrium value using minimax
                    q_matrix = q_values_new_state_target[i].reshape(self.num_actions_per_agent, self.num_actions_per_agent)
                    nash_value = compute_minimax_value(q_matrix)
                    target_value = rewards[i] + self.gamma * nash_value
                
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                targets[i][joint_index] = target_value
        else:
            # Standard Nash-Q
            q_values_new_state = self.brain.predict(new_states, target=True)
            
            for i in range(batch_size):
                if dones[i]:
                    target_value = rewards[i]
                else:
                    # NASH-Q: Use Nash equilibrium value instead of max Q
                    q_matrix = q_values_new_state[i].reshape(self.num_actions_per_agent, self.num_actions_per_agent)
                    nash_value = compute_minimax_value(q_matrix)
                    target_value = rewards[i] + self.gamma * nash_value
                
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                targets[i][joint_index] = target_value
        
        return targets

    def save_model(self):
        self.brain.save_model()
