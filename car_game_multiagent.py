"""
Car Game Multi-Agent Script

This script trains two DQN agents to play the car game (zero-sum).
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from environments.car_game.env import CarGame
from dqn_agent import Agent
import glob

ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'grid_size']


def get_name_brain(args, idx):
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return './results_car_game/weights_files/' + file_name_str + '_' + str(idx) + '.pth'


def get_name_rewards(args):
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return './results_car_game/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):
    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])
    return './results_car_game/timesteps_files/' + file_name_str + '.csv'


class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)
        self.env = CarGame(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']

    def run(self, agents, file1, file2):
        import time
        total_step = 0
        rewards_list = []
        timesteps_list = []
        losses_list = [[], []]  # Track losses for both agents
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting training for {self.episodes_number} episodes")
        print(f"{'='*60}\n")
        
        for episode_num in range(self.episodes_number):
            state = self.env.reset()
            
            if self.render:
                self.env.render()

            state = np.array(state)
            if state.ndim == 1:
                state = state.ravel()

            done = False
            reward_all = 0
            time_step = 0
            
            while not done and time_step < self.max_ts:
                # Get actions from both agents
                actions = []
                for agent in agents:
                    actions.append(agent.greedy_actor(state))
                
                # Take step in environment
                next_state, reward, done = self.env.step(actions)
                
                next_state = np.array(next_state)
                if next_state.ndim == 1:
                    next_state = next_state.ravel()

                # Store experience for both agents
                # Player 1 gets reward, Player 2 gets -reward (zero-sum)
                if not self.test:
                    agents[0].observe((state, actions, reward, next_state, done))
                    agents[1].observe((state, actions, -reward, next_state, done))  # Zero-sum
                    
                    for agent in agents:
                        agent.decay_epsilon()
                        if total_step >= self.filling_steps:
                            if total_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)
            
            # Track average loss for this episode for each agent
            for agent_idx, agent in enumerate(agents):
                if len(agent.losses) > 0:
                    avg_loss = np.mean(agent.losses[-time_step:]) if len(agent.losses) >= time_step else np.mean(agent.losses)
                    losses_list[agent_idx].append(avg_loss)
                else:
                    losses_list[agent_idx].append(0.0)

            # Progress reporting with ETA
            progress = (episode_num + 1) / self.episodes_number * 100
            elapsed = time.time() - start_time
            if episode_num > 0:
                eta = elapsed / (episode_num + 1) * (self.episodes_number - episode_num - 1)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                print(f"[{progress:5.1f}%] Episode {episode_num+1}/{self.episodes_number} | "
                      f"Steps: {time_step:3d} | Reward: {reward_all:7.2f} | "
                      f"ETA: {eta_min}m {eta_sec}s")
            else:
                print(f"[{progress:5.1f}%] Episode {episode_num+1}/{self.episodes_number} | "
                      f"Steps: {time_step:3d} | Reward: {reward_all:7.2f}")

            # Show final state if rendering is enabled
            if self.render and done:
                self.env.render()

        # Save results
        if not self.test:
            print(f"\n{'='*60}")
            print(f"Training completed in {int(elapsed//60)}m {int(elapsed%60)}s")
            print(f"Saving results...")
            
            df = pd.DataFrame(rewards_list, columns=['reward'])
            df.to_csv(file1, index=False)
            df = pd.DataFrame(timesteps_list, columns=['timesteps'])
            df.to_csv(file2, index=False)
            
            # Save losses
            loss_file_a = file1.replace('rewards_files', 'losses_files').replace('.csv', '_agent_a.csv')
            loss_file_b = file1.replace('rewards_files', 'losses_files').replace('.csv', '_agent_b.csv')
            df_loss_a = pd.DataFrame(losses_list[0], columns=['loss'])
            df_loss_a.to_csv(loss_file_a, index=False)
            df_loss_b = pd.DataFrame(losses_list[1], columns=['loss'])
            df_loss_b.to_csv(loss_file_b, index=False)
            
            for agent in agents:
                agent.brain.save_model()
            
            print(f"Results saved to:")
            print(f"  - {file1}")
            print(f"  - {file2}")
            print(f"  - {loss_file_a}")
            print(f"  - {loss_file_b}")
            print(f"{'='*60}\n")
            
            # Plot results
            self._plot_results(rewards_list, losses_list, timesteps_list)
    
    def _plot_results(self, rewards, losses, timesteps):
        """Generate and save plots of training results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        episodes = range(1, len(rewards) + 1)
        
        # Plot 1: Rewards
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= 10:
            window = min(50, len(rewards) // 10)
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax1.plot(episodes, smoothed, color='darkblue', linewidth=2, label=f'Smoothed ({window} ep)')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Car A Rewards per Episode', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Car A)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss for Car A
        ax2.plot(episodes, losses[0], alpha=0.3, color='red', label='Raw')
        if len(losses[0]) >= 10:
            window = min(50, len(losses[0]) // 10)
            smoothed = pd.Series(losses[0]).rolling(window=window, min_periods=1).mean()
            ax2.plot(episodes, smoothed, color='darkred', linewidth=2, label=f'Smoothed ({window} ep)')
        ax2.set_title('Car A (Crasher) Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss for Car B
        ax3.plot(episodes, losses[1], alpha=0.3, color='green', label='Raw')
        if len(losses[1]) >= 10:
            window = min(50, len(losses[1]) // 10)
            smoothed = pd.Series(losses[1]).rolling(window=window, min_periods=1).mean()
            ax3.plot(episodes, smoothed, color='darkgreen', linewidth=2, label=f'Smoothed ({window} ep)')
        ax3.set_title('Car B (Victim) Loss', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Episode Length
        ax4.plot(episodes, timesteps, alpha=0.3, color='purple', label='Raw')
        if len(timesteps) >= 10:
            window = min(50, len(timesteps) // 10)
            smoothed = pd.Series(timesteps).rolling(window=window, min_periods=1).mean()
            ax4.plot(episodes, smoothed, color='indigo', linewidth=2, label=f'Smoothed ({window} ep)')
        ax4.set_title('Episode Length (Steps)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_file = './results_car_game/figures/training_results.png'
        os.makedirs('./results_car_game/figures', exist_ok=True)
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_file}")
        
        # Show plot
        plt.show(block=False)
        plt.pause(3)  # Display for 3 seconds
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training Parameters
    parser.add_argument('-e', '--episode-number', default=5000, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.00025, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp', type=str, help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=10000, type=int, help='Number of steps between updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=1000000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=int, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DQN', type=str)
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER', type=str)
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture')
    parser.add_argument('-gn', '--gpu-num', default='0', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable test phase')

    # Game Parameters
    parser.add_argument('-g', '--grid-size', default=5, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=100, type=int, help='Maximum number of timesteps per episode')

    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_false', help='Turn on visualization')

    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']

    # Create results directories
    os.makedirs('./results_car_game/weights_files', exist_ok=True)
    os.makedirs('./results_car_game/rewards_files', exist_ok=True)
    os.makedirs('./results_car_game/timesteps_files', exist_ok=True)
    os.makedirs('./results_car_game/losses_files', exist_ok=True)

    env = Environment(args)
    state_size = env.env.state_size
    action_space = env.env.action_space()

    # Create two agents (one for each player)
    all_agents = []
    for player_idx in range(2):
        brain_file = get_name_brain(args, player_idx)
        all_agents.append(Agent(state_size, action_space, player_idx, brain_file, args))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)

    env.run(all_agents, rewards_file, timesteps_file)
