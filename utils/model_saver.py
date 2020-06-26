import torch
import os


class ModelSaver:
    def __init__(self, base_path, model, optim):
        self.base_path = base_path + '/'
        self.model = model
        self.optim = optim

    def save(self, step):
        checkpoint = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict()
        }
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        prefix = self.base_path + self.model.model_name + '_'
        ck_path = prefix + '_step_%d.pth' % step
        torch.save(checkpoint, ck_path)
        return ck_path
