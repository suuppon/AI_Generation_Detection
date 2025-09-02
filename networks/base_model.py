# from pix2pix
import os
import torch
import torch.nn as nn
from torch.nn import init

class BaseModel(nn.Module):
    """
    self.model: 실제 분류기(혹은 멀티타워 래퍼)를 붙여서 사용
    optimizer는 외부에서 붙일 수 있음. save/load 시 optimizer, total_steps도 함께 저장 가능.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')
        self.model = None
        self.optimizer = None

    def attach(self, model: nn.Module, optimizer=None):
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def save_networks(self, epoch):
        os.makedirs(self.save_dir, exist_ok=True)
        save_filename = f'model_epoch_{epoch}.pth'
        save_path = os.path.join(self.save_dir, save_filename)

        payload = {
            "model": self.model.state_dict() if self.model is not None else None,
            "optimizer": self.optimizer.state_dict() if (self.isTrain and self.optimizer is not None) else None,
            "total_steps": self.total_steps,
            "lr": self.lr,
        }
        torch.save(payload, save_path)
        print(f'[BaseModel] Saved: {save_path}')

    def load_networks(self, epoch):
        load_filename = f'model_epoch_{epoch}.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        print(f'[BaseModel] Loading from {load_path}')
        state = torch.load(load_path, map_location=self.device)

        # 두 형태 모두 지원: 1) payload dict  2) 바로 state_dict
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"])
            self.total_steps = state.get("total_steps", 0)
            self.lr = state.get("lr", self.opt.lr)
            if self.isTrain and (not getattr(self.opt, "new_optim", False)) and self.optimizer is not None:
                if state.get("optimizer") is not None:
                    self.optimizer.load_state_dict(state["optimizer"])
                    # 옵티마 state를 GPU로
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.opt.lr
                    for s in self.optimizer.state.values():
                        for k, v in s.items():
                            if torch.is_tensor(v):
                                s[k] = v.to(self.device)
        else:
            # 과거 형식: 모델 state_dict만 저장된 경우
            self.model.load_state_dict(state)

    def eval(self):  # noqa: A003
        self.model.eval()

    def train(self):  # noqa: A003
        self.model.train()

    @torch.no_grad()
    def test(self, *args, **kwargs):
        self.model.eval()
        return self.model(*args, **kwargs)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'init method [{init_type}] not implemented')
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print(f'initialize network with {init_type}')
    net.apply(init_func)