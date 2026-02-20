"""
Nash-Q Agent - Uses joint Q-values for multi-agent learning.

Key difference from DQN Agent:
- Single shared Q-function outputting Q(s, a_A, a_B) for all action pairs
- Will use Nash equilibrium computation for action selection (Phase 2-4)
"""

import random
import numpy as np
from nash_brain import NashBrain


class NashAgent(object):
    """Agent for Nash-Q Learning."""

    def __init__(self, state_size, num_actions_per_agent, num_agents, arguments, agent_id=0, weights_path=''):
        self.state_size = state_size
        self.num_actions_per_agent = num_actions_per_agent
        self.num_agents = num_agents
        self.joint_action_size = num_actions_per_agent ** num_agents  # 5^2 = 25
        
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
        
        # Step counter
        self.steps = 0

    def greedy_actor(self, state):
        """
        Phase 1: Use epsilon-greedy on JOINT actions.
        Later (Phase 4): Will use Nash equilibrium policy.
        
        For now: Pick random joint action or best joint action.
        """
        if np.random.rand() <= self.epsilon:
            # Random joint action
            joint_action_index = random.randrange(self.joint_action_size)
        else:
            # Greedy joint action
            q_values = self.brain.predict_one_sample(state)
            joint_action_index = np.argmax(q_values)
        
        # Convert joint action index to individual actions
        # For 2 agents with 5 actions each: joint_index = a_A * 5 + a_B
        action_A = joint_action_index // self.num_actions_per_agent
        action_B = joint_action_index % self.num_actions_per_agent
        
        return action_A, action_B
    
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
        # Package experience as tuple
        sample = (state, action_A, action_B, reward, new_state, done)
        
        if self.experience_replay_type == 'PER':
            # Compute TD error for prioritization
            q_values = self.brain.predict_one_sample(state)
            q_values_new = self.brain.predict_one_sample(new_state, target=True)
            
            joint_index = self.joint_action_to_index(action_A, action_B)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(q_values_new)
            
            error = abs(target - q_values[joint_index])
            self.memory.remember(sample, error)
        else:
            # UER just stores the sample
            self.memory.remember(sample)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        """Train on a batch of experiences."""
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
        
        # Track loss
        self.losses.append(self.brain.last_loss)
        self.episode_losses.append(self.brain.last_loss)  # Also track per episode
        
        # Update PER priorities
        if self.experience_replay_type == 'PER':
            predictions = self.brain.predict(states, target=False)
            td_errors = []
            for i in range(len(states)):
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                td_error = abs(targets[i][joint_index] - predictions[i][joint_index])
                td_errors.append(td_error)
            self.memory.update(batch_indices, td_errors)
        
        # Update step counter
        self.steps += 1
        
        # Update target network
        if self.steps % self.update_target_frequency == 0:
            self.brain.update_target_model()

    def find_targets(self, states, actions_A, actions_B, rewards, new_states, dones):
        """
        Phase 1: Use standard max Q-value for next state.
        Phase 3: Will replace with Nash equilibrium value.
        """
        batch_size = len(states)
        targets = self.brain.predict(states, target=False)
        
        if self.double_dqn:
            # Double DQN: use online network to select action, target network to evaluate
            q_values_new_state = self.brain.predict(new_states, target=False)
            q_values_new_state_target = self.brain.predict(new_states, target=True)
            
            for i in range(batch_size):
                if dones[i]:
                    target_value = rewards[i]
                else:
                    # Select best joint action using online network
                    best_joint_index = np.argmax(q_values_new_state[i])
                    # Evaluate using target network
                    target_value = rewards[i] + self.gamma * q_values_new_state_target[i][best_joint_index]
                
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                targets[i][joint_index] = target_value
        else:
            # Standard DQN
            q_values_new_state = self.brain.predict(new_states, target=True)
            
            for i in range(batch_size):
                if dones[i]:
                    target_value = rewards[i]
                else:
                    # Use max Q-value over all joint actions
                    target_value = rewards[i] + self.gamma * np.max(q_values_new_state[i])
                
                joint_index = self.joint_action_to_index(actions_A[i], actions_B[i])
                targets[i][joint_index] = target_value
        
        return targets

    def save_model(self):
        self.brain.save_model()
