from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_apha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 

        # if len(obs.shape) > 1:
        #     observation = obs
        # else:
        #     observation = obs[None]
        # observation = ptu.from_numpy(observation)
        # action_distribution = self(observation)
        # if sample:
        #     action = action_distribution.rsample()
        #     print("action_1", action.shape)
        # else:
        #     action = action_distribution.mean
        #     # action = action[None]
        #     print("action_2", action.shape)
        # action =  ptu.to_numpy(action)
        # return action, action_distribution

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        if sample:
            action = action_distribution.rsample()
        else:
            action = action_distribution.mean
        action =  ptu.to_numpy(action)
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            
            batch_logstd = self.logstd.clip(min = self.log_std_bounds[0], 
                                            max = self.log_std_bounds[1])

            batch_dim = batch_mean.shape[0]
            batch_logstd = batch_logstd.repeat(batch_dim, 1)

            # print("batch_mean", batch_mean.shape)
            # print("batch_logstd", batch_logstd.shape)
            # print("batch_mean", batch_mean,shape)

            action_distribution = sac_utils.SquashedNormal(
                batch_mean,
                torch.exp(batch_logstd),
            )
            return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value


        target_entropy = self.target_entropy

        obs = ptu.from_numpy(obs)
        pi = self(obs)
        action =  pi.rsample() 
        # action, pi = self.get_action(obs)
        # action = ptu.from_numpy(action)
        log_prob = pi.log_prob(action).sum(dim = -1)

        # obs = ptu.from_numpy(obs)
        # two_q = critic(obs, action)
        # q = torch.min(two_q, dim = -1).values
        q1, q2 = critic(obs, action)
        q = torch.min(q1, q2)

        alpha = torch.exp(self.log_alpha)
        alpha = alpha.detach()

        # check dim
        cd1 = log_prob.shape == q.shape
        assert(cd1, "dim mismatch")


        loss_ac = torch.mean(alpha * log_prob - q)
        self.optimizer.zero_grad()
        loss_ac.backward()
        self.optimizer.step()


        alpha = torch.exp(self.log_alpha)
        log_prob = log_prob.detach()
        # q = q.detach()

        loss_al = torch.mean(- alpha * log_prob - alpha * target_entropy)
        self.log_alpha_optimizer.zero_grad()
        loss_al.backward()
        self.log_alpha_optimizer.step()

        actor_loss = ptu.to_numpy(loss_ac)
        alpha_loss = ptu.to_numpy(loss_al)

        return actor_loss, alpha_loss, alpha