
import numpy as np
import matplotlib.pyplot as plt
import gym, copy

import torch
import torch.nn.functional as F

ENV = 'CartPole-v0'
GAMMA = 0.9
NUM_EPISODES = 1000

NUM_PROCESSES = 16 # 多环境
NUM_ADVANCED_STEP = 5 # 现在+未来步数

# NUM_ADVANCED_STEP=1： 882
# NUM_ADVANCED_STEP=2： 559
# NUM_ADVANCED_STEP=3： 529
# NUM_ADVANCED_STEP=4： 257
# NUM_ADVANCED_STEP=5： 174
# NUM_ADVANCED_STEP=10：92

critic_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5


class ExperienceReplay(object):
    # [o0, o1, o2]
    #     [a1, a2]
    #     [r1, r2]
    # [rt0, rt1, rt2]
    def __init__(self, num_steps, num_processes, num_state):
        #  [位置，速度，角度，角速度]
        # [num_steps+1, num_processes, 4]
        self.observations = torch.zeros(num_steps + 1, num_processes, num_state)
        # [num_steps+1, num_processes, 1]
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # 奖励值: [num_steps, num_processes, 1]
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        # 【左，右】: [num_steps, num_processes, 1]
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        # 各步之后未来奖励之和: [num_steps+1, num_processes, 1]
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        # 游标
        self.index = 0
    # 插入新(s,a,r)
    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index+1].copy_(current_obs)
        self.masks[self.index+1].copy_(mask)
        self.rewards[self.index] = reward
        self.actions[self.index] = action

        self.index = (self.index+1) % NUM_ADVANCED_STEP
    # 重新开始，旧状态放入
    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.rewards[step] + self.returns[step+1]*GAMMA*self.masks[step+1]


class ActorCritic(torch.nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super(ActorCritic, self).__init__()
        # dim_in -> dim_middle -> dim_middle
        self.state_MLP = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_middle),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_middle, dim_middle),
            torch.nn.ReLU()
        )
        self.actor = torch.nn.Linear(dim_middle, dim_out)
        self.critic = torch.nn.Linear(dim_middle, 1)

    def forward(self, x):
        state = self.state_MLP(x)
        return self.critic(state), self.actor(state)

    def act(self, x):
        actor_output = self.actor(self.state_MLP(x))
        # [batch, dim_out]
        action_probs = F.softmax(actor_output, dim=1)
        # [batch, 1]
        action = action_probs.multinomial(num_samples=1)
        return action

    def get_value(self, x):
        return self.critic(self.state_MLP(x))

    def evaluate_actions(self, x, actions):
        critic_output, actor_output = self.forward(x)
        # [batch, dim_out]
        log_probs = F.log_softmax(actor_output, dim=1)
        # [batch, 1]
        action_log_probs = log_probs.gather(1, actions)
        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()
        # [batch, 1], [batch, 1], x
        return critic_output, action_log_probs, entropy

class Opt(object):
    def __init__(self, actorcritic):
        self.actorcritic = actorcritic
        self.optimizer = torch.optim.Adam(self.actorcritic.parameters(), lr=0.01)
    def update(self, experiencereplay):
        # [batch*, 4], [batch*, 1]
        critic_output, action_log_probs, entropy = self.actorcritic.evaluate_actions(
            experiencereplay.observations[:-1].view(-1,4), experiencereplay.actions.view(-1,1))
        critic_output = critic_output.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
        action_log_probs = action_log_probs.view(NUM_ADVANCED_STEP, NUM_PROCESSES, 1)
        # [NUM_ADVANCED_STEP, NUM_PROCESSES, 1]
        advantages = experiencereplay.returns[:-1] - critic_output
        critic_loss = torch.pow(advantages,2).mean()

        action_gain = (action_log_probs*advantages.detach()).mean()
        # critic: V(s), actor: \pi_(a|s)是softmax概率, Q(s,a)
        # action_gain = log\pi_(a|s) (Q(s,a) - V(s))
        # critic_loss = (Q(s,a) - V(s))^2
        # entropy = \sum_a \pi_(a|s) log \pi_(a|s)
        # 注意：三个损失都是mean！
        total_loss = (critic_loss_coef*critic_loss - action_gain - entropy_coef*entropy)
        # 这里从实验结果来看，不稳定
        # 默认critic_loss_coef=0.5 entropy_coef=0.01
        # 1.1  entropy_coef = 0.01，收敛，198
        # 1.2  entropy_coef = 0.1，不收敛 (执行多遍)，极少部分收敛
        # 1.3  entropy_coef = 0.001，非常快收敛，117
        # 1.3  entropy_coef = 0.0，非常快收敛，145
        # 2 去掉-action_gain影响很大，不收敛
        # 3.1 critic_loss_coef=1,   403
        # 3.2 critic_loss_coef=0.5, 236
        # 3.3 critic_loss_coef=0.1, 211
        # 3.4 critic_loss_coef=0.01, 364
        # 3.5 critic_loss_coef=0.0, 不收敛 (执行多遍)

        self.actorcritic.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), max_grad_norm)
        self.optimizer.step()


class Environment:
    def run(self):
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]
        # 状态维度=4, 动作维度=2
        dim_in, dim_out = envs[0].observation_space.shape[0], envs[0].action_space.n
        dim_middle = 32
        actor_critic = ActorCritic(dim_in, dim_middle, dim_out)
        opt = Opt(actor_critic)

        experiencereplay = ExperienceReplay(NUM_ADVANCED_STEP, NUM_PROCESSES, dim_in)
        # [NUM_PROCESSES, dim_in]
        current_obs = torch.from_numpy(np.array([envs[i].reset() for i in range(NUM_PROCESSES)])).float()
        experiencereplay.observations[0].copy_(current_obs)

        each_steps = np.zeros([NUM_PROCESSES, 1])
        episode_rewards = np.zeros([NUM_PROCESSES, 1])
        final_rewards = np.zeros([NUM_PROCESSES, 1])

        for j in range(NUM_EPISODES):
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    # [NUM_PROCESSES, 1]: actor
                    actions = actor_critic.act(experiencereplay.observations[step]).squeeze(1).numpy()

                obs_np, rewards_np, done_np = np.zeros([NUM_PROCESSES, dim_in]), np.zeros([NUM_PROCESSES, 1]), np.zeros([NUM_PROCESSES, 1])
                for i in range(NUM_PROCESSES):
                    obs_np[i], rewards_np[i], done_np[i], _ = envs[i].step(actions[i])
                    # print(j, i, obs_np[i], rewards_np[i], done_np[i], _)
                    if done_np[i]: # 失败即结束
                        if each_steps[i]<195: # 中途倒下结束
                            rewards_np[i] = -1.
                        else:
                            rewards_np[i] = 1.
                        each_steps[i] = 0
                        obs_np[i] = envs[i].reset()
                    else:
                        rewards_np[i] = 0.
                        each_steps[i] += 1
                # 正在进行=1，否则0
                masks = np.array([[0.0] if done else [1.0] for done in done_np])

                episode_rewards += rewards_np
                # 正在进行保持原样，结束的重置
                final_rewards = final_rewards * masks + episode_rewards * (1 - masks)
                episode_rewards *= masks

                experiencereplay.insert(torch.from_numpy(obs_np*masks).float(),
                                        torch.from_numpy(actions).long().view(-1,1),
                                            torch.from_numpy(rewards_np).float(),
                                        torch.from_numpy(masks).float())

            with torch.no_grad():
                critic_output = actor_critic.get_value(experiencereplay.observations[-1].detach())
            experiencereplay.compute_returns(critic_output)

            opt.update(experiencereplay)
            experiencereplay.after_update()

            print(j, final_rewards.squeeze(-1))

            if final_rewards.sum() >= NUM_PROCESSES:
                print('连续成功！')
                break

cartpole_env = Environment()
cartpole_env.run()

