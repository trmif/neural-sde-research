# models/torchsde_model.py
import torch
import torch.nn as nn
import torchsde


class EulerSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, d, model_drift, variational_drift, diffusion, method="euler"):
        super().__init__()
        self.d = d
        self.method = method
        self.model_drift = model_drift
        self.variational_drift = variational_drift
        self.diffusion = diffusion

    def f(self, t, x):
        return self.model_drift(t, x) + self.variational_drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)

    def forward(self, batch_size, n_steps):
        x0 = torch.zeros(batch_size, self.d)
        ts = torch.linspace(0, 1, n_steps + 1)
        xs = torchsde.sdeint(self, x0, ts, method=self.method, dt=1.0 / n_steps)
        return xs[-1]