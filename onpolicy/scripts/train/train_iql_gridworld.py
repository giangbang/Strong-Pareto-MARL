# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from onpolicy.envs.gridworld import GridworldEnv

from torch.utils.tensorboard import SummaryWriter
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape  # type: ignore[assignment]

        self.action_dim = 1
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype





def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Gridworld":
                
                env = GridworldEnv(all_args.plan, separated_rewards=all_args.seperated_rewards)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            # has no effect
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Gridworld":
                env = GridworldEnv(all_args.plan, separated_rewards=all_args.seperated_rewards)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])



from onpolicy.config import get_config

parser = get_config()

def parse_args(parser):
    parser.add_argument('--plan', type=int,
                        default=1, help="name of the plan to run on")

    all_args = parser.parse_known_args()[0]

    return all_args

all_args = parse_args(parser)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = all_args.seed
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = all_args.cuda
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Gridworld"
    """the id of the environment"""
    plan: int = all_args.plan
    total_timesteps: int = all_args.num_env_steps
    """total timesteps of the experiments"""
    learning_rate: float = all_args.lr
    """the learning rate of the optimizer"""
    num_envs: int = all_args.n_rollout_threads
    """the number of parallel game environments"""
    buffer_size: int = 200000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = .2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 5
    """the frequency of training"""
    eval_epsl: float = 0.01
    """epsilon exploration used when evaluating"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space[0].shape).prod(), all_args.hidden_size),
            nn.ReLU(),
            nn.Linear(all_args.hidden_size, all_args.hidden_size),
            nn.ReLU(),
            nn.Linear(all_args.hidden_size, env.action_space[0].n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def eval(q_networks, eval_envs, tmp_env, writer, global_step):
    all_returns = []
    rewards = []
    eval_obses, eval_share_obs, eval_available_actions = eval_envs.reset()
    assert eval_obses.shape[0] == 1
    total_num_envs_steps = all_args.eval_episodes * all_args.episode_length
    for episode in range(total_num_envs_steps):
        actions = []
        for agent_id, q_network in zip(range(tmp_env.n_agents), q_networks):

            epsilon = args.eval_epsl
            if random.random() < epsilon:
                action = np.array([tmp_env.action_space[agent_id].sample() for _ in range(args.num_envs)])
            else:
                obs = eval_obses[:, agent_id]
                # print(obs[0][:9].reshape(3, 3)* 6)
                q_values = q_network(torch.Tensor(obs).to(device))
                action = torch.argmax(q_values, dim=1).cpu().numpy()
            actions.append(action)
        # before transpose: n_agent x n_threads
        # after transpose: n_thread x n_agent
        actions = np.array(actions).transpose()
        assert actions.shape == (1, tmp_env.n_agents)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obses, _, reward, dones, infos, _  = eval_envs.step(actions)
        reward = reward.squeeze(-1)
        rewards.append(reward)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if dones.all():
            # print(global_step)
            # nstep x n_thread x n_agent 
            returns = np.array(rewards).sum(axis=0).mean(axis=0)
            rewards = []
            all_returns.append(returns)

        eval_obses = copy.deepcopy(next_obses)


    all_returns = np.array(all_returns)
    print("eval average episode rewards of agent: " + str(np.mean(all_returns)))
    writer.add_scalar("eval_episode_rewards", np.mean(all_returns), global_step)
    for agent_id in range(tmp_env.n_agents):
        print(f"eval_episode_rewards_agent{agent_id}", np.mean(all_returns[:, agent_id]))
        writer.add_scalar(f"eval_episode_rewards_agent{agent_id}", np.mean(all_returns[:, agent_id]), global_step)
                


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = Args
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}/plan{args.plan}/iql/{all_args.experiment_name}/seed{args.seed}_{int(time.time())}/logs"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    tmp_env = GridworldEnv(all_args.plan, separated_rewards=all_args.seperated_rewards, seed=all_args.seed)

    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_networks = [QNetwork(tmp_env).to(device) for _ in range(tmp_env.n_agents)]
    optimizers = [optim.Adam(q_network.parameters(), lr=args.learning_rate) for q_network in q_networks]
    target_networks = [QNetwork(tmp_env).to(device) for _ in range(tmp_env.n_agents)]
    for target_network, q_network in zip(target_networks, q_networks):
        target_network.load_state_dict(q_network.state_dict())


    rbs = [ReplayBuffer(
        args.buffer_size,
        tmp_env.observation_space[0],
        tmp_env.action_space[0],
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs
    ) for _ in range(tmp_env.n_agents)]
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obses, _, _ = envs.reset()
    print("obses shape", obses.shape) # n_thread x n_agent x obs_shape
    episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads
    cnt_eps = 0
    rewards = []
    returns = None
    for global_step in range(1, all_args.num_env_steps+10, args.num_envs):
        # ALGO LOGIC: put action logic here
        actions = []
        for agent_id, q_network in zip(range(tmp_env.n_agents), q_networks):

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                action = np.array([tmp_env.action_space[agent_id].sample() for _ in range(args.num_envs)])
            else:
                obs = obses[:, agent_id]
                # print(obs[0][:9].reshape(3, 3)* 6)
                q_values = q_network(torch.Tensor(obs).to(device))
                action = torch.argmax(q_values, dim=1).cpu().numpy()
            actions.append(action)
        # before transpose: n_agent x n_threads
        # after transpose: n_thread x n_agent
        actions = np.array(actions).transpose()
        assert actions.shape == (args.num_envs, tmp_env.n_agents)
        # print(actions)
        # ac_0 = actions[..., 0]
        # actions[..., 0] = actions[..., 1]
        # actions[..., 1] = ac_0

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obses, _, reward, dones, infos, _  = envs.step(actions)
        reward = reward.squeeze(-1)
        # print(reward.shape)
        # reward = np.sum(reward, axis=-1)
        # reward = [reward for _ in range(tmp_env.n_agents)]
        # reward = np.array(reward).transpose()
        # print(reward)
        
        # reward = [reward for _ in range()]
        # next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # n_threads x n_agent 
        # reward = np.array(reward).squeeze(-1)
        rewards.append(reward)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if dones.all():
            # print(global_step)
            # nstep x n_thread x n_agent 
            returns = np.array(rewards).sum(axis=0).mean(axis=0)
            rewards = []
            cnt_eps += args.num_envs

        for agent_id, rb in enumerate(rbs):
            rb.add(obses[:, agent_id], next_obses[:, agent_id], 
                   actions[:, agent_id], 
                   reward[:, agent_id], 
                   dones[:, agent_id], 
                   {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obses = copy.deepcopy(next_obses)

        # log information
        if global_step % (all_args.log_interval * all_args.episode_length) == 0 and returns is not None:
            end = time.time()
            print("\n Plan {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(all_args.plan,
                            all_args.algorithm_name,
                            run_name,
                            cnt_eps,
                            episodes,
                            global_step,
                            all_args.num_env_steps,
                            int(global_step / (end - start_time))))

            train_infos = [{} for _ in range(tmp_env.n_agents)]
            for agent_id in range(tmp_env.n_agents):
                train_infos[agent_id].update({"average_episode_rewards": np.mean(returns[agent_id])})
                print("average episode rewards of agent{} is {}".format(agent_id, train_infos[agent_id]["average_episode_rewards"]))
            print("Avg rewards all agents:", np.mean(returns))
            for agent_id in range(tmp_env.n_agents):
                for k, v in train_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    writer.add_scalar(agent_k, v, global_step)
            writer.add_scalar("average_episode_rewards", np.mean(returns), global_step)


        if global_step % (all_args.eval_interval * all_args.episode_length * 10) == 0 and all_args.use_eval:
            eval(q_networks, eval_envs, tmp_env, writer, global_step)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                for agent_id, rb, optimizer, target_network, q_network in zip(range(tmp_env.n_agents), 
                                            rbs, optimizers, target_networks, q_networks):
                    # if agent_id == 1: continue
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 5000 == 0:
                        writer.add_scalar(f"agent{agent_id}/losses/td_loss", loss, global_step)
                        writer.add_scalar(f"agent{agent_id}/losses/q_values", old_val.mean().item(), global_step)
                        # print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for q_network, target_network in zip(q_networks, target_networks):
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
    if all_args.use_eval:
        eval(q_networks, eval_envs, tmp_env, writer, global_step)
    envs.close()
    writer.close()