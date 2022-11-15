from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch



class PCModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.loss = nn.MSELoss()
        self.f_hat = ptu.build_mlp(
            self.ob_dim,
            self.output_size,
            n_layers=self.n_layers,
            size=self.size,
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



    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        y_hat = self.f_hat(ob_no)
        y = self.f(ob_no).detach()
        pred_error = torch.sqrt((y_hat - y)**2).sum(1)

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
