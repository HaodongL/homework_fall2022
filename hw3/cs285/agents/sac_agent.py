from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure import sac_utils
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # two_q = self.critic_target(next_ob_no, ac_na)
        # q = torch.min(two_q, dim = -1).values
        # print("two_q", two_q[0:10,:])
        q1, q2 = self.critic_target(next_ob_no, ac_na)
        q = torch.min(q1, q2)
        alpha =  torch.exp(self.actor.log_alpha)


        pi = self.actor(next_ob_no)
        action =  pi.sample()
        # action, pi = self.actor.get_action(ptu.to_numpy(next_ob_no))
        # action = ptu.from_numpy(action)
        log_prob = pi.log_prob(action).sum(dim = -1)
        # log_prob = log_prob.detach()

        # print("action", action.shape)
        # print("log_prob", log_prob.shape)
        # print("q1", q1.shape)
        # print("re_n", re_n.shape)

        y = re_n + self.gamma * (1 - terminal_n) * (q - alpha * log_prob)
        y = y.detach()

        # y_hat = torch.min(self.critic(ob_no, ac_na), dim = -1).values
        y_hat1, y_hat2 = self.critic(ob_no, ac_na)


        # check dim
        cd1 = re_n.shape == log_prob.shape == q.shape == y.shape == y_hat1.shape
        cd2 = ac_na.shape == action.shape
        assert cd1 and cd2, "dim mismatch"


        # print("imput3 size", y_hat1.shape)
        # print("targ3 size", y.shape)
        # print("targ3 size", y,shape)

        # loss = self.critic.loss(y_hat, y)
        loss = torch.mean(self.critic.loss(y_hat1, y) + self.critic.loss(y_hat2, y))
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()

        critic_loss = ptu.to_numpy(loss)

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging
        n_c = self.agent_params['num_critic_updates_per_agent_update']
        n_a = self.agent_params['num_actor_updates_per_agent_update']

        targ_freq = self.critic_target_update_frequency

        loss_c = []
        loss_ac = []
        loss_al = []
        tem = []

        for i in range(n_c):
            loss_c = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            # loss_c.append(critic_loss)
            if i % targ_freq == 0:
                sac_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        for j in range(n_a):
            loss_ac, loss_al, _ = self.actor.update(ob_no, self.critic)
            tem = torch.exp(self.actor.log_alpha)
            # loss_ac.append(actor_loss)
            # loss_al.append(alpha_loss)
            # tem.append(torch.exp(self.actor.log_alpha))


        loss = OrderedDict()
        loss['Critic_Loss'] = loss_c
        loss['Actor_Loss'] = loss_ac
        loss['Alpha_Loss'] = loss_al
        loss['Temperature'] = tem

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
