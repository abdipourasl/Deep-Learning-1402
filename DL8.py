import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import numpy as np
import gym


def create_model(number_observation_features: int, number_actions: int) -> nn.Module:
    """Create the MLP model"""

    hidden_layer_features = 32

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer_features,
                  out_features=number_actions),
    )


def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:
    """Get the policy from the model, for a specific observation"""

    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    logits = model(observation_tensor)

    # Categorical will also normalize the logits for us
    return Categorical(logits=logits)


def get_action(policy: Categorical) -> tuple[int, torch.Tensor]:
    """Sample an action from the policy"""

    action = policy.sample()  # Unit tensor

    # Converts to an int, as this is what Gym environments require
    action_int = int(action.item())

    # Calculate the log probability of the action, which is required for
    # calculating the loss later
    log_probability_action = policy.log_prob(action)

    return action_int, log_probability_action


def calculate_loss(epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> torch.Tensor:
    """Calculate the 'loss' required to get the policy gradient"""

    return -(epoch_log_probability_actions * epoch_action_rewards).mean()


def train_one_epoch(
        env: gym.Env,
        model: nn.Module,
        optimizer: Optimizer,
        max_timesteps=5000,
        episode_timesteps=200) -> float:
    """Train the model for one epoch"""

    epoch_total_timesteps = 0
    epoch_returns: list[float] = []
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    # Loop through episodes
    while True:
        if epoch_total_timesteps > max_timesteps:
            break

        episode_reward: float = 0
        observation = env.reset()

        for timestep in range(episode_timesteps):
            env.render()  # Render the environment

            epoch_total_timesteps += 1

            policy = get_policy(model, observation)
            action, log_probability_action = get_action(policy)
            observation, reward, done, _ = env.step(action)

            episode_reward += reward
            epoch_log_probability_actions.append(log_probability_action)

            if done is True:
                for _ in range(timestep + 1):
                    epoch_action_rewards.append(episode_reward)
                break

        epoch_returns.append(episode_reward)

    epoch_loss = calculate_loss(torch.stack(
        epoch_log_probability_actions),
        torch.as_tensor(
        epoch_action_rewards, dtype=torch.float32)
    )

    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(np.mean(epoch_returns))


def train(epochs=40) -> None:
    """Train a Vanilla Policy Gradient model on CartPole"""

    env = gym.make('CartPole-v0')

    torch.manual_seed(0)
    env.seed(0)

    number_observation_features = env.observation_space.shape[0]
    number_actions = env.action_space.n
    model = create_model(number_observation_features, number_actions)

    optimizer = Adam(model.parameters(), 1e-2)

    for epoch in range(epochs):
        average_return = train_one_epoch(env, model, optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))

    env.close()  # Close the environment when done


if name == 'main':
    train()