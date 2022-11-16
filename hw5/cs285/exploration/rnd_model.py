from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch


def init_method_1(model):
    model.weight.data.uniform_(-1.73, 1.73)
    model.bias.data.uniform_(-1.73, 1.73)

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()



class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.huber_delta = hparams['huber_delta']

        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.loss = nn.MSELoss()
        self.f_hat = ptu.build_mlp(
            self.ob_dim,
            self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method = init_method_1
        )
        self.f = ptu.build_mlp(
            self.ob_dim,
            self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method = init_method_2
        )
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f_hat.to(ptu.device)
        self.f.to(ptu.device)


    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        y_hat = self.f_hat(ob_no)
        y = self.f(ob_no).detach()
        
        if self.huber_delta != float("inf"):
            f_huber = nn.HuberLoss(reduction='none', delta=self.huber_delta)
            pred_error = f_huber(y_hat, y).sum(1)
        else:
            pred_error = torch.sqrt(((y_hat - y)**2).sum(1))
            # pred_error = ((y_hat - y)**2).mean(1)

        # print('ob_no: ', ob_no.shape)
        # print('y_hat: ', y_hat.shape)
        # print('y: ', y.shape)
        # print('pred_error: ', pred_error.shape)
        # print('ob_no: ', ob_no,shape)
        return pred_error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learning_rate_scheduler.step()

        return {'Training Loss': ptu.to_numpy(loss)}
