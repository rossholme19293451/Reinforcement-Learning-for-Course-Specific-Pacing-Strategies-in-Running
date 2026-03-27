import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network with shared feature extraction.
    Outputs both policy (actor) and value (critic) function estimates.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        #shared feature extraction network, 2 hidden layers each with 256 units
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        #separate heads for policy and value function
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

        #Kaiming initialisation for ReLU network
        for layer in list(self.net) + [self.policy_head, self.value_head]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)

        #learnable log standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim) * 1.0)

    def forward(self, x):
        h = self.net(x)
        return self.policy_head(h), self.value_head(h)

    def get_action(self, obs):
        """
        Sample an action from the policy and compute log probability.
        Uses tanh squashing to bound actions to [-1, 1].
        """
        mean, value = self.forward(obs)

        #clamp std deviation
        std = torch.clamp(self.log_std.exp(), 0.001, 1.0)
        normal = torch.distributions.Normal(mean, std)

        #sample from Gaussian and apply tanh
        z = normal.rsample()
        action = torch.tanh(z)

        #compute log probability
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        return action, log_prob, value

class PPO_Agent:
    """
    Proximal Policy Optimization (PPO) agent for continuous control.
    Implements PPO with clipped surrogate objective and GAE.
    """
    def __init__(
            self,
            env_fn,
            device,
            frames_per_batch = 18_000,
            total_frames = 1_440_000,
            gamma = 0.99,  #discount factor
            lam = 0.95,  #GAE lambda parameter
            clip_epsilon = 0.2,  #PPO clipping parameter
            lr = 0.0005,
            epochs = 5,  #number of optimisation epochs per batch
            minibatch_size = 128,
    ):
        self.env = env_fn
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = device

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)

        #adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def collect_batch(self):
        """
        Collect a batch to experience by running the current policy in the environment.
        Returns trajectories of observations, actions, rewards, etc.
        """
        obs_list = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        obs, _ = self.env.reset()

        #collect a fixed number of environment steps
        for _ in range(self.frames_per_batch):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_t)

            action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            #store transition
            obs_list.append(obs)
            actions.append(action_np)
            log_probs.append(log_prob.cpu().item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().item())

            obs = next_obs
            if done:
                obs, _ = self.env.reset()

        #bootstrap value for advantage calculation
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.model(obs_t)[1].cpu().item()
        values.append(last_value)

        return (np.array(obs_list),
                np.array(actions),
                np.array(log_probs),
                np.array(rewards),
                np.array(dones),
                np.array(values).reshape(-1),
                )

    def compute_gae(self, rewards, values, dones):
        """
        Compute the Generalised Advantage Estimate (GAE) for PPO.
        Balances bias-variance tradeoff in advantage estimates using lambda param.
        """
        adv = np.zeros_like(rewards)
        last_gae_lam = 0

        #backward pass through trajectory to compute advantages.
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

            #TD error: reward + discounted next value - current value
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            #GAE accumulation with exponential weighting
            adv[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        returns = adv + values[:-1]
        return adv, returns

    def train(self):
        """
        Main training loop: collect batches, compute advantages, and optimise the policy.
        """
        frame_count = 0
        training_log = []

        while frame_count < self.total_frames:
            #collect batches of experience
            (   obs_list,
                actions,
                old_log_probs,
                rewards,
                dones,
                values,
            ) = self.collect_batch()
            old_values = values[:-1]

            #compute advantages using GAE
            advantages, returns = self.compute_gae(rewards, values, dones)

            #normalise advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            #multiple epochs of optimisation on collected batch
            dataset_size = len(obs_list)
            for epoch in range(self.epochs):
                idxs = np.arange(dataset_size)
                np.random.shuffle(idxs)

                #process in mini-batches
                for start in range(0, dataset_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    batch_idx = idxs[start:end]

                    #convert batch to tensors
                    obs_t = torch.tensor(obs_list[batch_idx], dtype=torch.float32).to(self.device)
                    actions_t = torch.tensor(actions[batch_idx], dtype=torch.float32).to(self.device)
                    old_log_probs_t = torch.tensor(old_log_probs[batch_idx], dtype=torch.float32).to(self.device)
                    adv_t = torch.tensor(advantages[batch_idx], dtype=torch.float32).to(self.device)
                    ret_t = torch.tensor(returns[batch_idx], dtype=torch.float32).to(self.device)
                    old_values_t = torch.tensor(old_values[batch_idx], dtype=torch.float32).to(self.device)

                    #forward pass with current policy
                    mean, value = self.model(obs_t)
                    value = value.squeeze()

                    std = torch.clamp(self.model.log_std.exp(), 0.001, 1.0)
                    normal = torch.distributions.Normal(mean, std)

                    #recompute log probabilities for collected actions
                    actions_clipped = torch.clamp(actions_t, -1.0 + 1e-6, 1.0 - 1e-6)

                    z = torch.atanh(actions_clipped)
                    new_log_prob = (normal.log_prob(z) - torch.log(1 - actions_t.pow(2) + 1e-6)).sum(-1)

                    #PPO clipped surrogate objective
                    ratio = (new_log_prob - old_log_probs_t).exp()
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    #clipped value function loss
                    value_clipped = old_values_t + torch.clamp(value - old_values_t, -self.clip_epsilon, self.clip_epsilon)
                    value_loss_unclipped = (ret_t - value).pow(2)
                    value_loss_clipped = (ret_t - value_clipped).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    #entropy bonus for exploration
                    entropy = normal.entropy().sum(-1).mean()

                    #combined loss: policy + value + entropy
                    loss = policy_loss + 0.5 * value_loss - 0.001 * entropy

                    #gradient descent step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            #print rewards from batch
            frame_count += self.frames_per_batch
            print(f"Frames: {frame_count}/{self.total_frames}")
            print(len(rewards), " ", rewards.sum())

            training_log.append({"Frame_Count": frame_count,
                                 "Rewards": rewards.sum()})

        #plot rewards
        df = pd.DataFrame(training_log)
        print(df.head())

        plt.figure(figsize = (10,5))
        plt.plot(df["Frame_Count"], df["Rewards"], label="Rewards")
        plt.xlabel("Frames")
        plt.ylabel("Total Rewards per Batch")
        plt.grid(True)
        plt.show()


    def run(self, episodes = 1):
        """
        Run the trained policy for evaluation and visualisation.
        Collects episode data for analysis.
        """
        all_episodes = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            episode_data = {
                "distance": [],
                "velocity": [],
                "energy": [],
                "action": [],
                "reward": [],
                "time": []
            }

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.model.get_action(obs_t)
                action = action.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = self.env.step(action)
                print(f"Force = {obs[4]}, Reward = {reward}")
                print(info)
                self.env.render()
                done = terminated or truncated
                total_reward += reward

                #store trajectory data for analysis
                episode_data["distance"].append(obs[0])
                episode_data["velocity"].append(obs[1])
                episode_data["energy"].append(obs[2])
                episode_data["action"].append(obs[4])
                episode_data["reward"].append(reward)
                episode_data["time"].append(info["time"])

            all_episodes.append(episode_data)
            print(f"Episode: {ep}, Reward: {total_reward}")

        return all_episodes