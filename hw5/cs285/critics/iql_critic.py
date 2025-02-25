from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np

from cs285.infrastructure import pytorch_util as ptu

class IQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        
        # TODO define value function
        # HINT: see Q_net definition above and optimizer below
        ### YOUR CODE HERE ###
        self.v_net = network_initializer(self.ob_dim, 1)

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler_v  = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile = hparams['iql_expectile']

    def expectile_loss(self, diff):
        """
        Implement expectile loss on the difference between q and v
        """
        e_input = self.iql_expectile - (diff < 0).type(torch.int64)
        e_loss = torch.mean(torch.abs(e_input)*(diff ** 2))

        # print("diff ", diff.shape)
        # print("e_input ", e_input.shape)
        # print("e_loss ", e_loss)
        # print("diff ", diff,shape)
        return e_loss

    def update_v(self, ob_no, ac_na):
        """
        Update value function using expectile loss
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        

        ### YOUR CODE HERE ###
        qa_vals = self.q_net_target(ob_no)
        q_vals = torch.gather(qa_vals, 1, ac_na.unsqueeze(1)).squeeze(1)
        v_vals = self.v_net(ob_no).squeeze(1)
        diff = q_vals - v_vals


        # print("diff ", diff.shape)
        # print("q_vals ", q_vals.shape)
        # print("qa_vals ", qa_vals.shape)
        # print("v_vals ", v_vals.shape)
        # print("diff ", diff,shape)
        value_loss = self.expectile_loss(diff)
        
        assert value_loss.shape == ()
        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()
        self.learning_rate_scheduler_v.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}


    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Use target v network to train Q
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        ### YOUR CODE HERE ###
        v_vals = self.v_net(next_ob_no).squeeze(1)
        target = reward_n + self.gamma * v_vals
        qa_vals = self.q_net(ob_no)
        q_vals = torch.gather(qa_vals, 1, ac_na.unsqueeze(1)).squeeze(1)
        loss = self.mse_loss(q_vals, target)

        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
