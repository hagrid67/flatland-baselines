import getopt
import random
import sys
import time
from typing import List

import numpy as np
import os

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params

from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.utils.ordered_set import OrderedSet

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet

train_set = True

if train_set: random_seed = np.random.randint(1000)
else: random_seed = np.random.randint(1000,2000)

test_env_no = np.random.randint(10)
level_no = np.random.randint(2)
random.seed(random_seed)
np.random.seed(random_seed)


class MultipleAgentNavigationObs(TreeObsForRailEnv):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self,max_depth: int, predictor: PredictionBuilder = None):
        super().__init__(max_depth,predictor)

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        distance_map = self.env.distance_map.get()

        visited = set()
        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited

        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(agent_virtual_position, direction)
                min_distances.append(
                    distance_map[handle, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        if num_transitions == 1:
            observation = [0, 1, 0]
            observation = np.tile(observation, 2)

        elif num_transitions == 2:
            idx = np.argpartition(np.array(min_distances), 2)
            observation1 = [0, 0, 0]
            observation1[idx[0]] = 1

            observation2 = [0, 0, 0]
            observation2[idx[1]] = 1

            observation = np.hstack([observation1, observation2])

        min_distances = np.sort(min_distances)
        incremental_distances = np.diff(np.sort(min_distances))
        incremental_distances[incremental_distances == np.inf] = -1
        incremental_distances[np.isnan(incremental_distances)] = -1
        min_distances[min_distances == np.inf] = -1
        observation = np.hstack([observation, incremental_distances[0]])

        distance_target = distance_map[(handle, *agent_virtual_position,
                                                            agent.direction)]
        observation = np.hstack([distance_target, observation,
                                                            agent.malfunction_data['malfunction'],
                                                            agent.speed_data['speed'],agent.speed_data['position_fraction']])

        return observation


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    sleep_for_animation = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        else:
            assert False, "unhandled option"

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
                       obs_builder_object=MultipleAgentNavigationObs(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30)))

    max_steps = int(4 * 2 * (20 + env.height + env.width))

    obs, info = env.reset(regenerate_rail=True,
            regenerate_schedule=True,
            activate_agents=False,
            random_seed=random_seed)
    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env(show=True, frames=True, show_observations=True)
    n_agents = env.get_num_agents()
    x_dim, y_dim = env.width,env.height
    # Reset score and done
    score = 0
    env_done = 0
    step = 0
    for step in range(max_steps):
        action_dict = {}
        for i in range(n_agents):
            if not obs:
                action_dict.update({i: 2})
            elif obs[i] is not None:
                action = np.argmax(obs[i][1:4]) + 1
                action_dict.update({i: action})

        obs, all_rewards, done, _ = env.step(action_dict)
        print("Rewards: ", all_rewards, "  [done=", done, "]")

        for a in range(env.get_num_agents()):
            score += all_rewards[a] / env.get_num_agents()

        env_renderer.render_env(show=True, frames=True, show_observations=True)
        if sleep_for_animation:
            time.sleep(0.5)
        if done["__all__"]:
            break

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window = tasks_finished / max(1, env.get_num_agents())
        scores_window = score / max_steps
        print(
            '\rTraining {} Agents on ({},{}).\t Steps {}\t Average Score: {:.3f}\tDones: {:.2f}%\t'.format(
                n_agents, x_dim, y_dim,
                step,
                np.mean(scores_window),
                100 * np.mean(done_window)), end=" ")

    tasks_finished = 0
    for current_agent in env.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1
    done_window = tasks_finished / max(1, env.get_num_agents())
    scores_window = score / max_steps
    print(
        '\rTraining {} Agents on ({},{}).\t Total Steps {}\t Average Score: {:.3f}\tDones: {:.2f}%\t'.format(
            n_agents, x_dim, y_dim,
            step,
            np.mean(scores_window),
            100 * np.mean(done_window)), end=" ")

    env_renderer.close_window()


if __name__ == '__main__':
    if 'argv' in globals():
        main(sys.argv)
    else:
        main(sys.argv[1:])
