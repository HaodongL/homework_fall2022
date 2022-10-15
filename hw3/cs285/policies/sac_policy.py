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

        if sample:
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]
            observation = ptu.from_numpy(observation)
            action_distribution = self(observation)
            action = action_distribution.rsample()
            action =  ptu.to_numpy(action)
        else:
            action = action_distribution.mean()
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
            
            self.logstd.clip(min = self.log_std_bounds[0], 
                             max = self.log_std_bounds[1])

            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = sac_utils.SquashedNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs = ptu.from_numpy(obs)
        pi = self(obs)
        action =  pi.rsample() 
        log_prob = pi.log_prob(action)


        two_q = critic(obs, action)
        q = torch.min(two_q, axis = 1)

        loss_ac = torch.mean(self.alpha * log_prob - q)
        self.optimizer.zero_grad()
        loss_ac.backward()
        self.optimizer.step()


        loss_al = torch.mean(- self.alpha * log_prob - self.alpha * self.target_entropy)
        self.optimizer.zero_grad()
        loss_al.backward()
        self.optimizer.step()

        actor_loss = ptu.to_numpy(loss_ac)
        alpha_loss = ptu.to_numpy(loss_al)

        return actor_loss, alpha_loss, self.alpha