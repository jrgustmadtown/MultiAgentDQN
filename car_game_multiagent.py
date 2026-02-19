"""
Car Game Multi-Agent Script

This script trains two DQN agents to play the car game (zero-sum).
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
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
        total_step = 0
        rewards_list = []
        timesteps_list = []
        
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

            print(f"Episode {episode_num}/{self.episodes_number}, Steps: {time_step}, Reward: {reward_all:.2f}")

            if self.render:
                self.env.render()

        # Save results
        if not self.test:
            df = pd.DataFrame(rewards_list, columns=['reward'])
            df.to_csv(file1, index=False)
            df = pd.DataFrame(timesteps_list, columns=['timesteps'])
            df.to_csv(file2, index=False)
            
            for agent in agents:
                agent.brain.save_model()


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
