import getopt
import random
import sys
from collections import deque
import time
# make sure the root path is in system path
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from utils.observation_utils import normalize_observation
from dueling_double_dqn import Agent

from flatland.envs.observations import TreeObsForRailEnv  # , TreeObsAdditionalForRailEnv, MultipleAgentNavigationObs
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

from flatland.envs.malfunction_generators import malfunction_from_params

from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file


def train_validate_env_generator_params(train_set, n_agents, x_dim, y_dim, observation, stochastic_data,
                                        speed_ration_map, seed=1):
    if train_set:
        random_seed = np.random.randint(1000)
    else:
        random_seed = np.random.randint(1000, 2000)
    random.seed(random_seed)
    np.random.seed(random_seed)

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=seed,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  obs_builder_object=observation)
    return env, random_seed


def train_validate_env_generator(train_set, observation):
    if train_set:
        random_seed = np.random.randint(1000)
    else:
        random_seed = np.random.randint(1000, 2000)

    test_env_no = np.random.randint(9)
    level_no = np.random.randint(2)
    random.seed(random_seed)
    np.random.seed(random_seed)

    test_envs_root = f"./test-envs/Test_{test_env_no}"
    test_env_file_path = f"Level_{level_no}.pkl"

    test_env_file_path = os.path.join(
        test_envs_root,
        test_env_file_path
    )
    print(f"Testing Environment: {test_env_file_path} with seed: {random_seed}")

    env = RailEnv(width=1, height=1, rail_generator=rail_from_file(test_env_file_path),
                  schedule_generator=schedule_from_file(test_env_file_path),
                  malfunction_generator_and_process_data=malfunction_from_file(test_env_file_path),
                  obs_builder_object=observation)
    return env, random_seed


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    random.seed(1)
    np.random.seed(1)

    training = True
    weights = False

    #  Setting these 2 parameters to True can slow down the network for training
    visuals = False
    sleep_for_animation = False

    trained_epochs = 1  # set to trained epoch number if resuming training or for validation , else value is 1

    if training:
        train_set = True
    else:
        train_set = False
        weights = True

    dataset_type = "MANUAL"  # set as MANUAL if we want to specify our own parameters, AUTO otherwise
    # Parameters for the Environment
    x_dim = 25
    y_dim = 25
    n_agents = 4

    # Use a the malfunction generator to break agents from time to time
    stochastic_data = {'prop_malfunction': 0.05,  # Percentage of defective agents
                       'malfunction_rate': 100,  # Rate of malfunction occurence
                       'min_duration': 20,  # Minimal duration of malfunction
                       'max_duration': 50  # Max duration of malfunction
                       }

    # Custom observation builder
    tree_observation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    if dataset_type == "MANUAL":
        env, random_seed = train_validate_env_generator_params(train_set, n_agents, x_dim, y_dim, tree_observation,
                                                               stochastic_data, speed_ration_map)
    else:
        env, random_seed = train_validate_env_generator(train_set, tree_observation)

    # Given the depth of the tree observation and the number of features per node we get the following state_size
    num_features_per_node = env.obs_builder.observation_dim
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes
    # state_size = state_size + n_agents*12
    # state_size = num_features_per_node
    # The action space of flatland is 5 discrete actions
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_trials' not in locals():
        n_trials = 15000

    # And the max number of steps we want to take per episode
    max_steps = int(4 * 2 * (20 + env.height + env.width))

    columns = ['Agents', 'X_DIM', 'Y_DIM', 'TRIAL_NO', 'SCORE',
               'DONE_RATIO', 'STEPS', 'ACTION_PROB']
    df_all_results = pd.DataFrame(columns=columns)

    # Define training parameters
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998

    # And some variables to keep track of the progress
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size

    # Now we load a Double dueling DQN agent
    agent = Agent(state_size, action_size)

    trial_start = 1
    if weights:
        trial_start = trained_epochs
        eps = max(eps_end, (np.power(eps_decay, trial_start)) * eps)
        weight_file = f"navigator_checkpoint{trial_start}.pth"
        with open("./Nets/" + weight_file, "rb") as file_in:
            agent.qnetwork_local.load_state_dict(torch.load(file_in))

    for trials in range(trial_start, n_trials + 1):

        if dataset_type == "MANUAL":
            env, random_seed = train_validate_env_generator_params(train_set, n_agents, x_dim, y_dim, tree_observation,
                                                                   stochastic_data, speed_ration_map)
        else:
            env, random_seed = train_validate_env_generator(train_set, tree_observation)
        # Reset environment
        obs, info = env.reset(regenerate_rail=True,
                              regenerate_schedule=True,
                              activate_agents=False,
                              random_seed=random_seed)

        env_renderer = RenderTool(env, gl="PILSVG", )
        if visuals:
            env_renderer.render_env(show=True, frames=True, show_observations=True)

        x_dim, y_dim, n_agents = env.width, env.height, env.get_num_agents()

        agent_obs = [None] * n_agents
        agent_next_obs = [None] * n_agents
        agent_obs_buffer = [None] * n_agents
        agent_action_buffer = [2] * n_agents
        cummulated_reward = np.zeros(n_agents)
        update_values = [False] * n_agents
        # Build agent specific observations
        for a in range(n_agents):
            # if obs[a] is not None:
            #     agent_obs[a] = obs[a]
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Reset score and done
        score = 0
        env_done = 0
        step = 0
        # Run episode
        while True:
            # Action
            for a in range(n_agents):
                if info['action_required'][a]:
                    # If an action is require, we want to store the obs a that step as well as the action
                    update_values[a] = True
                    action = agent.act(agent_obs[a], eps=eps)
                    action_prob[action] += 1
                else:
                    update_values[a] = False
                    action = 0
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            step += 1
            if visuals:
                env_renderer.render_env(show=True, frames=True, show_observations=True)
                if sleep_for_animation:
                    time.sleep(0.5)
            for a in range(n_agents):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values[a] or done[a]:
                    agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                               agent_obs[a], done[a])
                    cummulated_reward[a] = 0.

                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if next_obs[a]:
                    agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)
                # if next_obs[a] is not None:
                #     agent_obs[a] = next_obs[a]
                score += all_rewards[a] / n_agents

            # Copy observation
            if done['__all__']:
                env_done = 1
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        data = [[n_agents, x_dim, y_dim,
                 trials,
                 np.mean(scores_window),
                 100 * np.mean(done_window),
                 step, action_prob / np.sum(action_prob)]]

        df_cur = pd.DataFrame(data, columns=columns)
        df_all_results = pd.concat([df_all_results, df_cur])

        df_all_results.to_csv(f'{dataset_type}_DQN_TrainingResults_{n_agents}_{x_dim}_{y_dim}.csv', index=False)

        if trials % 100 == 0:
            print(
                '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(), x_dim, y_dim,
                    trials,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps, action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './Nets/navigator_checkpoint' + str(trials) + '.pth')
            action_prob = [1] * action_size

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
