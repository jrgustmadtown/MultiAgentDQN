"""
Nash-Q Brain - Joint Action-Value Function

Neural network that outputs Q(s, a_A, a_B) for all action pairs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

HUBER_LOSS_DELTA = 1.0


class JointDQN(nn.Module):
    """Network that outputs Q-values for all joint actions."""
    def __init__(self, state_size, num_actions_per_agent, num_agents, num_nodes):
        super(JointDQN, self).__init__()
        self.num_actions_per_agent = num_actions_per_agent
        self.num_agents = num_agents
        self.joint_action_size = num_actions_per_agent ** num_agents  # 4^2 = 16
        
        self.fc1 = nn.Linear(state_size, num_nodes)
        self.fc2 = nn.Linear(num_nodes, num_nodes)
        self.fc3 = nn.Linear(num_nodes, self.joint_action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output: [batch, 25] for all (a_A, a_B) pairs
        return x


class JointDuelingDQN(nn.Module):
    """Dueling architecture for joint Q-values."""
    def __init__(self, state_size, num_actions_per_agent, num_agents, num_nodes):
        super(JointDuelingDQN, self).__init__()
        self.num_actions_per_agent = num_actions_per_agent
        self.num_agents = num_agents
        self.joint_action_size = num_actions_per_agent ** num_agents
        
        # Value stream
        self.value_fc1 = nn.Linear(state_size, num_nodes)
        self.value_fc2 = nn.Linear(num_nodes, num_nodes)
        self.value_fc3 = nn.Linear(num_nodes, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(state_size, num_nodes)
        self.advantage_fc2 = nn.Linear(num_nodes, num_nodes)
        self.advantage_fc3 = nn.Linear(num_nodes, self.joint_action_size)
        
    def forward(self, x):
        # Value stream
        value = torch.relu(self.value_fc1(x))
        value = torch.relu(self.value_fc2(value))
        value = self.value_fc3(value)
        
        # Advantage stream
        advantage = torch.relu(self.advantage_fc1(x))
        advantage = torch.relu(self.advantage_fc2(advantage))
        advantage = self.advantage_fc3(advantage)
        
        # Combine: Q(s,a,b) = V(s) + (A(s,a,b) - mean(A(s,a,b)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = torch.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * torch.square(err)
    L1 = HUBER_LOSS_DELTA * (torch.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = torch.where(cond, L2, L1)
    return torch.mean(loss)


class NashBrain(object):
    """Brain for Nash-Q Learning with joint Q-values."""

    def __init__(self, state_size, num_actions_per_agent, num_agents, brain_name, arguments):
        self.state_size = state_size
        self.num_actions_per_agent = num_actions_per_agent
        self.num_agents = num_agents
        self.joint_action_size = num_actions_per_agent ** num_agents  # 5^2 = 25
        
        self.weight_backup = brain_name.replace('.h5', '.pth')
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.dueling = arguments['dueling']
        self.optimizer_model = arguments['optimizer']
        self.gradient_clip = arguments.get('gradient_clip', 1.0)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build models
        self.model = self._build_model()
        self.model_ = self._build_model()
        self.model_.load_state_dict(self.model.state_dict())
        
        # Setup optimizer
        if self.optimizer_model == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_model == 'RMSProp':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            print('Invalid optimizer!')
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Track loss for visualization
        self.last_loss = 0.0
        
        # Load weights if in test mode
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error: no file')
            else:
                self.model.load_state_dict(torch.load(self.weight_backup, map_location=self.device))
                self.model_.load_state_dict(torch.load(self.weight_backup, map_location=self.device))

    def _build_model(self):
        if self.dueling:
            model = JointDuelingDQN(self.state_size, self.num_actions_per_agent, 
                                   self.num_agents, self.num_nodes)
        else:
            model = JointDQN(self.state_size, self.num_actions_per_agent, 
                           self.num_agents, self.num_nodes)
        
        model.to(self.device)
        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):
        self.model.train()
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(x_tensor)
            
            if sample_weight is not None:
                sample_weight_tensor = torch.FloatTensor(sample_weight).to(self.device)
                loss = huber_loss(y_tensor, predictions)
                loss = (loss * sample_weight_tensor).mean()
            else:
                loss = huber_loss(y_tensor, predictions)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            self.last_loss = loss.item()

    def predict(self, state, target=False):
        """
        Predict Q-values for all joint actions.
        
        Returns:
            Q-values: [batch_size, 25] array of Q(s, a_A, a_B)
        """
        model = self.model_ if target else self.model
        model.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            predictions = model(state_tensor)
            return predictions.cpu().numpy()

    def predict_one_sample(self, state, target=False):
        """Predict Q-values for a single state."""
        return self.predict(state.reshape(1, self.state_size), target=target).flatten()
    
    def get_q_matrix(self, state, target=False):
        """
        Get Q-values as a matrix for game-theoretic analysis.
        
        Returns:
            Q_matrix: [4, 4] array where Q_matrix[a_A, a_B] = Q(s, a_A, a_B)
        """
        q_values = self.predict_one_sample(state, target=target)
        # Reshape from [16] to [4, 4]
        return q_values.reshape(self.num_actions_per_agent, self.num_actions_per_agent)

    def update_target_model(self):
        self.model_.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), self.weight_backup)
