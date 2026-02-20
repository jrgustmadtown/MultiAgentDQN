"""
Nash-Q Training Script for Car Game

Phase 1: Joint Action-Value Representation
- Single agent with Q(s, a_A, a_B) for all 25 action pairs
- Standard DQN learning for now (will add Nash equilibrium in Phase 3-4)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from environments.car_game.env import CarGame
from nash_agent import NashAgent


class Environment(object):
    """Wrapper for Nash-Q training."""

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)
        self.env = CarGame(arguments, current_path)
        
        self.num_agents = 2
        self.num_actions_per_agent = 5
        self.state_size = self.env.state_size
        
        # Results directory
        self.results_path = os.path.join(current_path, 'results_nashq_car_game')
        self.weights_path = os.path.join(self.results_path, 'weights_files')
        self.rewards_path = os.path.join(self.results_path, 'rewards_files')
        self.timesteps_path = os.path.join(self.results_path, 'timesteps_files')
        self.losses_path = os.path.join(self.results_path, 'losses_files')
        self.figures_path = os.path.join(self.results_path, 'figures')
        
        # Create directories
        for path in [self.results_path, self.weights_path, self.rewards_path, 
                     self.timesteps_path, self.losses_path, self.figures_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.episode_count = arguments['episode_count']
        self.render = arguments['render']
        self.test = arguments['test']
        
        # Single agent for Nash-Q
        self.agent = NashAgent(self.state_size, self.num_actions_per_agent, 
                               self.num_agents, arguments, agent_id=0, 
                               weights_path=self.weights_path)
        
        # Tracking
        self.rewards_list = []
        self.timesteps_list = []
        self.losses_list = []

    def run(self):
        """Main training/testing loop."""
        print("\n" + "="*60)
        print("NASH-Q LEARNING - Phase 1: Joint Action-Value Representation")
        print("="*60)
        print(f"Environment: Car Game ({self.env.grid_size}x{self.env.grid_size} grid)")
        print(f"State size: {self.state_size}")
        print(f"Actions per agent: {self.num_actions_per_agent}")
        print(f"Joint action space: {self.num_actions_per_agent**2}")
        print(f"Mode: {'TEST' if self.test else 'TRAIN'}")
        print(f"Episodes: {self.episode_count}")
        print("="*60 + "\n")
        
        start_time = time()
        
        for episode_num in range(self.episode_count):
            episode_start = time()
            
            # Reset episode tracking
            self.agent.episode_losses = []  # Track losses for this episode only
            
            state = self.env.reset()
            done = False
            episode_reward = 0
            timesteps = 0
            
            while not done:
                if self.render:
                    self.env.render()
                
                # Get joint action from Nash agent
                action_A, action_B = self.agent.greedy_actor(state)
                
                # Execute in environment
                new_state, reward, done = self.env.step([action_A, action_B])
                
                # Store experience and learn
                if not self.test:
                    self.agent.observe(state, action_A, action_B, reward, new_state, done)
                    self.agent.replay()
                
                state = new_state
                episode_reward += reward
                timesteps += 1
            
            # Decay exploration
            if not self.test:
                self.agent.decay_epsilon()
            
            # Track metrics
            self.rewards_list.append(episode_reward)
            self.timesteps_list.append(timesteps)
            
            # Track loss from THIS episode only
            if not self.test and len(self.agent.episode_losses) > 0:
                avg_loss = np.mean(self.agent.episode_losses)
            else:
                avg_loss = 0
            self.losses_list.append(avg_loss)
            
            # Progress update
            episode_time = time() - episode_start
            elapsed_time = time() - start_time
            
            if (episode_num + 1) % 10 == 0 or self.test:
                avg_reward = np.mean(self.rewards_list[-10:])
                avg_timesteps = np.mean(self.timesteps_list[-10:])
                avg_loss = np.mean(self.losses_list[-10:]) if len(self.losses_list) > 0 else 0
                
                eps_remaining = self.episode_count - (episode_num + 1)
                eta_seconds = eps_remaining * (elapsed_time / (episode_num + 1))
                eta_minutes = eta_seconds / 60
                
                print(f"Ep {episode_num+1:4d}/{self.episode_count} | "
                      f"R: {episode_reward:6.1f} (avg: {avg_reward:6.1f}) | "
                      f"Steps: {timesteps:3d} (avg: {avg_timesteps:5.1f}) | "
                      f"Loss: {avg_loss:6.4f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"ETA: {eta_minutes:.1f}m")
            
            # Save periodically
            if not self.test and (episode_num + 1) % 100 == 0:
                self.agent.save_model()
                self._save_metrics()
        
        # Final save
        if not self.test:
            self.agent.save_model()
            self._save_metrics()
            self._plot_results()
        
        total_time = time() - start_time
        print(f"\n{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Average reward: {np.mean(self.rewards_list):.2f}")
        print(f"Average timesteps: {np.mean(self.timesteps_list):.2f}")
        if not self.test:
            print(f"Average loss: {np.mean(self.losses_list):.4f}")
            print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print(f"{'='*60}\n")

    def _save_metrics(self):
        """Save rewards, timesteps, and losses to CSV."""
        base_name = 'nashq_car_game'
        
        # Rewards
        rewards_file = os.path.join(self.rewards_path, f'{base_name}.csv')
        pd.DataFrame({'episode': range(len(self.rewards_list)), 
                     'reward': self.rewards_list}).to_csv(rewards_file, index=False)
        
        # Timesteps
        timesteps_file = os.path.join(self.timesteps_path, f'{base_name}.csv')
        pd.DataFrame({'episode': range(len(self.timesteps_list)), 
                     'timesteps': self.timesteps_list}).to_csv(timesteps_file, index=False)
        
        # Losses
        losses_file = os.path.join(self.losses_path, f'{base_name}.csv')
        pd.DataFrame({'episode': range(len(self.losses_list)), 
                     'loss': self.losses_list}).to_csv(losses_file, index=False)

    def _plot_results(self):
        """Generate training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Nash-Q Learning - Phase 1 Training Results', fontsize=16)
        
        episodes = range(len(self.rewards_list))
        
        # Plot 1: Rewards
        axes[0, 0].plot(episodes, self.rewards_list, alpha=0.3, color='blue')
        if len(self.rewards_list) >= 10:
            window = 10
            smoothed = pd.Series(self.rewards_list).rolling(window=window).mean()
            axes[0, 0].plot(episodes, smoothed, color='blue', linewidth=2, label=f'{window}-episode avg')
            axes[0, 0].legend()
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Losses
        axes[0, 1].plot(episodes, self.losses_list, alpha=0.3, color='red')
        if len(self.losses_list) >= 10:
            window = 10
            smoothed = pd.Series(self.losses_list).rolling(window=window).mean()
            axes[0, 1].plot(episodes, smoothed, color='red', linewidth=2, label=f'{window}-episode avg')
            axes[0, 1].legend()
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss (Huber)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Episode Length
        axes[1, 0].plot(episodes, self.timesteps_list, alpha=0.3, color='green')
        if len(self.timesteps_list) >= 10:
            window = 10
            smoothed = pd.Series(self.timesteps_list).rolling(window=window).mean()
            axes[1, 0].plot(episodes, smoothed, color='green', linewidth=2, label=f'{window}-episode avg')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Exploration
        # Reconstruct epsilon decay
        epsilon_values = []
        epsilon = arguments['epsilon_start']
        for _ in range(len(self.rewards_list)):
            epsilon_values.append(epsilon)
            if epsilon > arguments['epsilon_stop']:
                epsilon *= arguments['epsilon_decay']
        
        axes[1, 1].plot(episodes, epsilon_values, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.figures_path, 'nashq_training_results.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_file}")
        plt.close()
        
        # Also generate loss diagnostic plot
        self._plot_loss_diagnostic()
    
    def _plot_loss_diagnostic(self):
        """Generate detailed loss analysis plot."""
        fig = plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_list, alpha=0.7, color='red')
        plt.title(f'Loss over {len(self.losses_list)} episodes')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        n = min(100, len(self.losses_list))
        plt.plot(self.losses_list[-n:], alpha=0.7, color='darkred')
        plt.title(f'Last {n} episodes')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        diagnostic_file = os.path.join(self.results_path, 'loss_diagnostic.png')
        plt.savefig(diagnostic_file, dpi=100)
        print(f"Loss diagnostic saved to: {diagnostic_file}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode-count', default=1000, type=int, help='Number of episodes')
    parser.add_argument('-r', '--render', action='store_true', help='Render environment')
    parser.add_argument('-test', '--test', action='store_true', help='Test mode')
    parser.add_argument('-g', '--grid-size', default=5, type=int, help='Grid size (must be odd)')
    
    args = parser.parse_args()
    
    arguments = {
        'grid_size': args.grid_size,
        'render': args.render,
        'test': args.test,
        'episode_count': args.episode_count,
        'max_timestep': 100,
        
        # Nash-Q parameters (tuned for 25-output network)
        'batch_size': 16,  # Smaller for speed
        'learning_rate': 1e-5,  # Lower for stability with joint Q-values
        'gamma': 0.99,
        'memory_capacity': 10000,  # Much smaller for speed
        'epsilon_start': 1.0,
        'epsilon_stop': 0.01,
        'epsilon_decay': 0.995,
        'update_target_frequency': 100,
        'number_nodes': 128,  # Smaller network for speed
        'dueling': False,  # Disable dueling for speed
        'double_dqn': True,
        'experience_replay': 'UER',  # Simpler for now, avoid PER overhead
        'pr_scale': 0.5,
        'optimizer': 'Adam',  # Adam often more stable than RMSProp
        'gradient_clip': 0.5,  # Stronger clipping
        'train_frequency': 1  # Train every step for better loss tracking
    }
    
    environment = Environment(arguments)
    environment.run()
