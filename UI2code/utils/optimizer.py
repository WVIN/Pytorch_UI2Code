from torch.nn.utils import clip_grad_norm_


class Optimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.training_step = 1

    def state_dict(self):
        return {
            'training_step': self.training_step,
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.training_step = state_dict['training_step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self, loss):
        loss.backward()

    def step(self):
        clip_grad_norm_(self.optimizer.param_groups[0]['params'], 20)
        self.optimizer.step()
        self.training_step += 1
